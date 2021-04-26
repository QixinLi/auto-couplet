# -*- coding: utf-8 -*-
import csv
import json
from copy import deepcopy

from transformers import BertTokenizer, RobertaTokenizer
from tqdm import tqdm


class InputFeature():
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "input_ids:{}\ninput_mask:{}\nsegment_ids:{}\nlabel_ids:{}".format(
            self.input_ids,
            self.input_mask,
            self.segment_ids,
            self.label_ids,
        )

class data_handler():
    def __init__(self, tokenizer: BertTokenizer, max_seq_length):
        self.domains = []
        # self.roberta = RobertaModel.from_pretrained('../roberta.base', checkpoint_file='model.pt')
        self.tokenizer = tokenizer
        self.utterances = []
        self.vocab_words = list(self.tokenizer.get_vocab().keys())
        self.max_seq_len = max_seq_length
        self.mask_mode = "normal"
        self.pad_token = self.tokenizer.pad_token
        self.pad_idx = -100
        self.total_convert = 0
        self.convert_success = 0
        self.cls = self.tokenizer.cls_token
        self.sep = self.tokenizer.sep_token
        self.input_pad = self.tokenizer.pad_token_id
        self.token_type_input_pad = self.tokenizer.pad_token_type_id
        self.mask_token = self.tokenizer.mask_token

    def convert_examples_to_features(self, dir):
        features = []
        f_in = open(f"{dir}/in.txt", "r")
        f_out = open(f"{dir}/out.txt", "r")
        in_examples = f_in.readlines()
        out_examples = f_out.readlines()
        assert len(in_examples) == len(out_examples)
        for i, o in zip(in_examples, out_examples):
            in_words = i.strip().split()
            out_words = o.strip().split()
            assert len(in_words) == len(out_words)
            features.append(self.convert_one_example_to_feature(in_words, out_words))
        return features

    def convert_one_example_to_feature(self, text_a: list, text_b: list):
        input_ids = []
        labels = []
        input_ids.append(self.tokenizer.cls_token_id)
        labels.append(self.pad_idx)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(text_a))
        labels.extend(self.tokenizer.convert_tokens_to_ids(text_b))
        assert len(input_ids) == len(labels)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            input_mask = input_mask[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            input_ids.extend([self.input_pad] * (self.max_seq_len - len(input_ids)))
            input_mask.extend([0] * (self.max_seq_len - len(input_mask)))
            labels.extend([self.pad_idx] * (self.max_seq_len - len(labels)))
            segment_ids.extend([self.token_type_input_pad] * (self.max_seq_len - len(segment_ids)))
        assert len(input_ids) == self.max_seq_len \
               and len(segment_ids) == self.max_seq_len \
               and len(input_mask) == self.max_seq_len \
               and len(labels) == self.max_seq_len
        item = InputFeature(
            input_ids=deepcopy(input_ids),
            input_mask=deepcopy(input_mask),
            segment_ids=deepcopy(segment_ids),
            label_ids=deepcopy(labels)
        )
        return item


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", do_lower_case=True)
    print(f"tokenizer.cls_token_id:{tokenizer.cls_token_id}")
    print(f"tokenizer.pad_token_id:{tokenizer.pad_token_id}")
    print(f"tokenizer.sep_token_id:{tokenizer.sep_token_id}")
    print(tokenizer.convert_ids_to_tokens([100]))
    handler = data_handler(
        tokenizer=tokenizer,
        max_seq_length=128
    )
    handler.convert_examples_to_features("./couplet/train")
    # handler.getAllData()
    # handler.generate_Pretraining_Data()
    # handler.split_training_data()
