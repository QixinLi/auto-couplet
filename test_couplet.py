# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from six.moves import xrange
import tensorflow as tf
from pyltp import Segmentor
import os
import sys

# 初始化LTP分词模型
LTP_DATA_DIR = 'H:\\Python\\NLP_Learning_Files\\LTP\\ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

vocabulary_size = 100000

data_index = 0

batch_size = 128
embedding_size = 128  
skip_window = 1       
num_skips = 2         
valid_size = 5      
valid_window = 100  
num_sampled = 64    

stuff = "空"

segmentor = Segmentor()
segmentor.load(cws_model_path)
tf.reset_default_graph()

def read_data():
    words = open("./dictionary/dic.txt", "r", encoding='utf-8')
    raw_word_list = []
    for line in words.readlines():
        raw_word_list.append(line.strip('\n'))
    return raw_word_list

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def getNearestWord(words,uplink):
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print("成功创建词典集")
    #删除words节省内存
    del words  

    valid_string = uplink
    valid_word = segmentor.segment(valid_string)
    valid_size=len(valid_word)
    try:
        valid_examples =[dictionary[li] for li in valid_word]
    except:
        try:
            valid_examples =[dictionary[li] for li in valid_string]
            valid_word = list(valid_string)
        except:
            print("所填词汇超出词典范围")
            return "NULL"
        
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
        # Add variable initializer.
        init = tf.global_variables_initializer()
    
    
    model_dir = './model'
    
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
        session.run(tf.global_variables_initializer()) # 先对模型初始化
        # We must initialize all variables before we use them.
        saver.restore(session, os.path.join(model_dir,'model.ckpt'))
        init.run()
        #print("Initialized")
        sim = similarity.eval()
        afterTransfer=""
        for i in xrange(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 50  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[:top_k]
            for k in xrange(top_k):
                # print(str(len(reverse_dictionary))+' '+str(nearest[k]))
                close_word = reverse_dictionary[nearest[k]]
                #print(close_word+valid_word)
                if close_word != valid_word and len(close_word)==len(valid_word):
                    afterTransfer+=close_word
                    break
                if k+1==top_k:
                    afterTransfer+=stuff*len(valid_word) # 字典中找不到的内容用空字符补全
        return afterTransfer
    return "NULL"

def main(uplink):

    if(len(uplink)<=1):
        uplink = "天打雷劈"  # 如果没有参数，则设置默认值
    else:
        uplink = uplink[1]
    words = read_data()
    print("成功读取数据")
    downlink = getNearestWord(words,uplink)
    
    if len(uplink)>len(downlink) :
        print("不足补全")
        downlink = downlink+stuff*(len(uplink)-len(downlink))
    
    print("上联："+uplink)
    print("下联："+downlink)
 
if __name__ == "__main__":
    main(sys.argv)
