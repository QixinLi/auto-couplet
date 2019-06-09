# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import numpy as np
from six.moves import xrange
import tensorflow as tf
from pyltp import Segmentor
import os

# 初始化LTP分词模型
LTP_DATA_DIR = 'H:\\Python\\NLP_Learning_Files\\LTP\\ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
tf.reset_default_graph()

def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    #读取停用词
    stop_words = []
    with open('stop_words.txt',"r",encoding="UTF-8") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    # 读取文本，预处理，分词，得到词典
    raw_word_list = []
    
    in_dialogs=[]
    out_dialogs=[]
    for line in open("dialog/in.txt", encoding="utf8",errors='ignore'):
        if line.strip():
            in_dialogs.append(line.replace(" ","").replace("\n",""))
    print("共载入"+str(len(in_dialogs))+"条上联")
    for line in open("dialog/out.txt", encoding="utf8",errors='ignore'):
        if line.strip():
            out_dialogs.append(line.replace(" ","").replace("\n",""))
    print("共载入"+str(len(out_dialogs))+"条下联")
    
    max_num = 200000 #自定义数据集大小
    
    print("选取其中"+str(max_num)+"条")
    in_dialogs = in_dialogs[:max_num]
    out_dialogs = out_dialogs[:max_num]
    segmentor = Segmentor()
    segmentor.load(cws_model_path)

    for i in range(min(len(in_dialogs),len(out_dialogs))):
        if len(in_dialogs[i])>0: # 如果句子非空
            raw_words = list(segmentor.segment(in_dialogs[i]))
            raw_word_list.extend(raw_words)
        if len(out_dialogs[i])>0: # 如果句子非空
            raw_words = list(segmentor.segment(out_dialogs[i]))
            raw_word_list.extend(raw_words)
    segmentor.release()
    return raw_word_list

def write_words_2_file(words):
    dic = open("./dictionary/dic.txt", "w", encoding='utf-8')
    for w in words:
        dic.write(w+"\n")
    dic.close()

#step 1:读取文件中的内容组成一个列表,并将词典保存至本地
words = read_data()
write_words_2_file(words)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count",len(count))
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

data, count, dictionary, reverse_dictionary = build_dataset(words)
#删除words节省内存
del words  

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128  
skip_window = 1       
num_skips = 2         
valid_size = 5
valid_window = 100  
num_sampled = 64    # Number of negative examples to sample.

segmentor = Segmentor()
segmentor.load(cws_model_path)

valid_string = "白日依山尽"
valid_word = segmentor.segment(valid_string)
valid_size=len(valid_word)
valid_examples =[dictionary[li] for li in valid_word]

graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]),dtype=tf.float32)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases, 
                                         inputs=embed, 
                                         labels=train_labels,
                                         num_sampled=num_sampled, 
                                         num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 3000000  #训练轮数



model_dir = './model'

with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
    session.run(tf.global_variables_initializer()) # 先对模型初始化
    # We must initialize all variables before we use them.
    saver.restore(session, os.path.join(model_dir,'model.ckpt'))
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values fstep / 10000or session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            print("save——"+str(step/10000))
            saver.save(session, os.path.join(model_dir,'model.ckpt'))
        
            
    final_embeddings = normalized_embeddings.eval()
    
    
