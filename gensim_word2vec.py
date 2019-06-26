# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:16:04 2019
@author: lee
"""
import sys
import socket
import os
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from pyltp import Segmentor

# 初始化LTP分词模型
LTP_DATA_DIR = 'H:\\Python\\NLP_Learning_Files\\LTP\\ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
stuff = "空"
segmentor = Segmentor()
segmentor.load(cws_model_path)

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim 

    wv_from_bin = gensim.models.KeyedVectors.load_word2vec_format('H:\\Python\\trained_model\\baike_26g_news_13g_novel_229g.bin',binary=True) 
    vocab = list(wv_from_bin.vocab.keys())
    print("预训练模型包含单词总数 %i" % len(vocab))
    return wv_from_bin

wv_from_bin = load_word2vec()


def get_downlink(data, isSeg=True):
    if isSeg:
        valid_word = segmentor.segment(data)
    else:
        valid_word = data
    print(list(valid_word))
    downlink = ""
    for i in range(len(valid_word)):
        v = valid_word[i]
        try:
            most_similar = wv_from_bin.most_similar(positive=[v])
        except:
            if not valid_word==data:
                print("分词超出词汇范围")
                downlink = get_downlink(data, False)
                break
            else:
                print("文本超出词汇范围")
                downlink = "文本超出词汇范围"
                break
        
        
        for i in range(len(most_similar)):
            if len(most_similar[i][0])==len(v):
                print(most_similar[i][0])
                downlink = downlink + most_similar[i][0]
                break
            if (i+1)==len(most_similar):
                num = 0
                j = 0
                d_temp = ""
                while (len(v) - num)>=len(most_similar[j][0]):
                    d_temp = d_temp + most_similar[j][0]
                    num = num+len(most_similar[j][0])
                    j = j+1
                d_temp = d_temp + most_similar[j][0][:(len(v) - num)]
                print(d_temp)
                downlink = downlink + d_temp
                
    print("返回下联:"+str(downlink))
    
    if not isinstance(downlink,bytes):
        downlink = downlink.encode()
    return downlink

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("localhost", 9201))
sock.listen(5)
print("打开本地socket，端口：9201")
while True:
    
    connection,address = sock.accept()
    data = connection.recv(1024)
    data = data.decode()
    
    if data == "exit":
        connection.send("exit successfully".encode())
        connection.close()
        break
    
    print("收到上联:"+data)
    
    downlink = get_downlink(data)
    
    connection.send(downlink)
    connection.close()


