# word2vec_auto-couplet
采用word2vec模型，让计算机也会对对联

## 前言

偶然看见github上的大牛们用seq2seq模型训练出自动对对联的程序，于是自己寻思换一个思路，用word2vec试试。

原理在于，给上联分词后的每个词语找到字典里欧氏距离最近的词（理论上的同义词），再将其组合成下联

其中

word2vec模型搭建参考 [tensorflow官方给出的源码](https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)

对中文进行词向量训练参考 [Deermini/word2vec-tensorflow](https://github.com/Deermini/word2vec-tensorflow)

数据集来源 [wb14123/couplet-dataset](https://github.com/wb14123/couplet-dataset)

LTP模型下载地址 [哈工大语言云百度网盘](https://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569#list/path=%2F)

中文word2vec预训练模型参考 [268G+训练好的word2vec模型（中文词向量）](https://www.jianshu.com/p/ae5b45e96dbf)

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

`gensim_word2vec.py`  加载训练集，并在本地打开socket端口监听

`gensim_test_couplet.py`  通过socket调用训练集并返回数据

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

```
成功读取数据
成功创建词典集
上联：两袖清风存正气
下联：一网情深萏褒封
```

每次测试结果都不一样，大家也可以试一试

## 更新日志

### 2019-6-26
- 利用gensim调用了[简书用户___dada____](https://www.jianshu.com/p/ae5b45e96dbf)训练好的word2vec模型

- 新增 `gensim_word2vec.py` ，加载训练集，并在本地打开socket端口监听

- 新增 `gensim_test_couplet.py` ，通过socket调用训练集并返回数据

### 2019-6-16
- 新增三国演义和唐诗宋词数据集

唐诗宋词数据集来自[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)

繁体汉字转简体调用[skydark/nstools](https://github.com/skydark/nstools/tree/master/zhtools)

- 新增 `transfer2hans.py` ，用来将唐诗宋词数据集中的古诗词整理成便于训练的 `唐诗宋词.txt`
- 优化数据存储结构