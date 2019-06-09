# word2vec_auto-couplet
采用word2vec模型，让计算机也会对对联

## 前言

偶然看见github上的大牛们用seq2seq模型训练出自动对对联的程序

于是自己寻思换一个思路

用word2vec试试


原理在于

给上联分词后的每个词语找到字典里欧氏距离最近的词（理论上的同义词），再将其组合成下联


其中

word2vec模型搭建参考 [tensorflow官方给出的源码](https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)

对中文进行词向量训练参考 [Deermini/word2vec-tensorflow](https://github.com/Deermini/word2vec-tensorflow)

数据集来源 [wb14123/couplet-dataset](https://github.com/wb14123/couplet-dataset)


## 开发环境
 - Python 3.5
 - Tensorflow
 - 哈工大LTP

## 项目结构
`./dialog` 中保存训练的数据 `in.txt` 和 `out.txt` 

`./dictionary` 中保存本地字典 `dic.txt`

`./model` 中保存训练的模型

`stop_words.txt` 中文文本停用词

`train_couplet.py` 训练脚本

`test_couplet.py` 测试脚本


## 使用
<li>训练

`
python train_couplet.py
`

<li>使用

`
python test_couplet.py 白日依山尽
`

## 测试结果
输入

`
python test_couplet.py 两袖清风存正气
`

输出

`
成功读取数据
成功创建词典集
上联：两袖清风存正气
下联：一网情深萏褒封
`

每次测试结果都不一样，大家也可以试一试