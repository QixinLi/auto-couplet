# -*- coding: utf-8 -*-
import random
import time

from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataset import TensorDataset
from transformers import AutoTokenizer
import os
import numpy as np
from torch.optim import AdamW
from argparse import ArgumentParser
from bert_model.modeling import BertForCouplet
from bert_model.utils import *
from bert_model.collate import *
import wandb

class Pretrainer():

    def __init__(self,args):
        self.is_wandb = False and not args.debug
        ## 超参数
        self.fp16 = args.fp16
        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.evaluate_per_epoch = args.evaluate_per_epoch
        self.weight_decay = 0.01
        self.adam_b1 = 0.9
        self.adam_b2 = 0.98
        self.adam_e = 1e-6
        self.warmup_proportion = 0.1
        self.max_seq_length = args.max_seq_length
        self.max_patience = args.patience
        # 路径相关
        self.saved_model_dir = "./save/"
        self.save_name = f"auto_v0.1_{self.lr}.bin"
        self.checkpoint_name = self.saved_model_dir + self.save_name

        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # self.model = AutoModelForMaskedLM.from_pretrained(self.plm_dir)
        self.model = BertForCouplet.from_pretrained("hfl/chinese-roberta-wwm-ext")

        self.gpus = list(map(int, args.gpu.split(",")))
        print("gpus: {}".format(self.gpus))
        self.device = torch.device("cuda:{}".format(self.gpus[0]) if torch.cuda.is_available() else "cpu")

        # =========train dataset===========
        dh = data_handler(tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        data_dir = "./couplet"
        train_features = dh.convert_examples_to_features(f"{data_dir}/train")
        test_features = dh.convert_examples_to_features(f"{data_dir}/test")
        self.train_dataloader = self.dataloader(train_features)
        self.test_dataloader = self.dataloader(test_features)

    def dataloader(self, features, is_train=True):
        input_ids = []
        input_mask = []
        segment_ids = []
        label_ids = []
        for feature in features:
            input_ids.append(feature.input_ids)
            input_mask.append(feature.input_mask)
            segment_ids.append(feature.segment_ids)
            label_ids.append(feature.label_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        params = [input_ids, input_mask, segment_ids, label_ids]
        dataset = TensorDataset(*params)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=is_train)

    def evaluate(self,dev_dataloader):
        sum_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(dev_dataloader)
            for batch in iterator:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, token_type_ids, label_ids = batch
                (loss, prediction_scores) = self.model(input_ids=input_ids, attention_mask=input_mask,token_type_ids=token_type_ids, labels=label_ids)
                sum_loss += float(loss.item())
        return sum_loss

    def train(self):
        self.model.to(self.device)
        # 优化器相关
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
        params = {'in_pretrained': {'decay': [], 'no_decay': []}, 'out_pretrained': {'decay': [], 'no_decay': []}}

        for n, p in param_optimizer:
            is_in_pretrained = 'in_pretrained' if 'roberta' in n else 'out_pretrained'
            is_no_decay = 'no_decay' if any(nd in n for nd in no_decay) else 'decay'
            params[is_in_pretrained][is_no_decay].append(p)

        grouped_parameters = [
            {'params': params['in_pretrained']['decay'], 'weight_decay': self.weight_decay, 'lr': self.lr},
            {'params': params['in_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': self.lr},
            {'params': params['out_pretrained']['decay'], 'weight_decay': self.weight_decay, 'lr': self.lr},
            {'params': params['out_pretrained']['no_decay'], 'weight_decay': self.weight_decay, 'lr': self.lr},
        ]

        optimizer = AdamW(
            grouped_parameters,
            lr=self.lr,
            # weight_decay=self.weight_decay,
            betas=(self.adam_b1, self.adam_b2),
            eps=self.adam_e
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex .")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O1")

        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)
        self.model.to(self.device)

        if self.is_wandb:
            this_time = time.strftime("%m%d-%H%M%S", time.localtime())
            wandb.init(project="x-pretrain", name=f'csmlm_plus_{this_time}')

        best_loss = float("inf")
        best_model_state_dict = None
        min_loss_unchanged_num = 0
        global_step = 0
        for i in range(self.epochs):
            iterator = tqdm(self.train_dataloader)
            for batch in iterator:
                global_step += 1
                self.model.train()
                optimizer.zero_grad()

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, token_type_ids, label_ids = batch

                (loss, prediction_scores) = self.model(input_ids=input_ids, attention_mask=input_mask,token_type_ids=token_type_ids, labels=label_ids)
                loss.backward()
                optimizer.step()
                iterator.set_description("Epoch {}, loss: {}".format(i, loss.cpu()))
                if self.is_wandb:
                    wandb.log({
                        "loss": loss.cpu()
                    })
                del loss, prediction_scores

            eval_loss = self.evaluate(self.test_dataloader)
            if best_loss > eval_loss:
                best_model_state_dict = deepcopy(self.model.state_dict())
                torch.save({'state_dict': best_model_state_dict}, self.checkpoint_name)
                min_loss_unchanged_num = 0
            else:
                min_loss_unchanged_num += 1
                if min_loss_unchanged_num >= self.max_patience:
                    torch.save({'state_dict': best_model_state_dict},self.checkpoint_name)
                    return

        torch.save({'state_dict': best_model_state_dict}, self.checkpoint_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=11111)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--evaluate_per_epoch', type=int, default=1)

    parser.add_argument("--fp16",action="store_true")
    parser.add_argument("--debug", action="store_true")
    # My add
    parser.add_argument("--patience", type=int, default=5, help="earlystop")
    parser.add_argument("--max_seq_length",default=64,type=int)
    args = parser.parse_args()
    print(args)
    ptm = Pretrainer(args)
    ptm.train()