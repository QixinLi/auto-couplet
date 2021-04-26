# -*- coding: utf-8 -*-
# @Author  : Qixin Li
# @Time    : 2021/4/26 下午4:42

import socket

from transformers import BertTokenizer
from bert_model.modeling import BertForCouplet
import torch


tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = BertForCouplet.from_pretrained("hfl/chinese-roberta-wwm-ext")

if torch.cuda.is_available():
    model_CKPT = torch.load("./save/auto_v0.1_5e-05.bin")  # TODO fill the model name
    model.load_state_dict(model_CKPT['state_dict'])
    model = model.cuda()
else:
    model_CKPT = torch.load("./save/auto_v0.1_5e-05.bin", map_location="cpu")  # TODO fill the model name
    model.load_state_dict(model_CKPT['state_dict'])


def predict(model: BertForCouplet, text: str):
    tokens = tokenizer.tokenize(text)
    input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask], dtype=torch.long)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
    (_, prediction_scores) = model(input_ids=input_ids, attention_mask=input_mask,token_type_ids=segment_ids)
    final_prediction_tensor = prediction_scores.argmax(dim=-1)
    if torch.cuda.is_available():
        final_prediction_list = final_prediction_tensor.cpu().numpy().tolist()[0]
    else:
        final_prediction_list = final_prediction_tensor.numpy().tolist()[0]
    result = tokenizer.convert_ids_to_tokens(final_prediction_list[1:])
    return "".join(result)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("localhost", 9201))
sock.listen(5)
print("打开本地socket，端口：9201")
connection, address = sock.accept()
print("已接入终端链接.....")
while True:
    data = connection.recv(1024)
    data = data.decode()

    if data == "exit":
        connection.send("exit successfully".encode())
        connection.close()
        break

    print("收到上联:" + data)

    downlink = predict(model, data)
    if not isinstance(downlink,bytes):
        downlink = downlink.encode()
    connection.send(downlink)
print("链接断开.....")
connection.close()
