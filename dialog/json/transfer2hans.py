# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:49:26 2019

@author: lee
"""

from langconv import *
import sys
import json

file = open('唐诗宋词.txt','w',encoding='utf-8')
file.seek(0)
file.truncate()
print("重置本地文件")

# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line



def transfer2hans(file_name):
    jsondata = open(file_name, encoding='UTF-8').read()
    poets = json.loads(jsondata)
    
    for i in range(len(poets)):
        poet = poets[i]['paragraphs']
        for j in range(len(poet)):
            file.write(cht_to_chs(poet[j]))
        file.write('\n')

def write_song_poet(num):
    file_head = 'poet.song.'
    for i in range(num+1):
        transfer2hans(file_head+str(i)+'000.json')
        if (i+1) % 10 == 0 or i == num:
            print("成功转化第"+str(i+1)+"个宋词文件")
def write_tang_poet(num):
    file_head = 'poet.tang.'
    for i in range(num+1):
        transfer2hans(file_head+str(i)+'000.json')
        if (i+1) % 10 == 0 or i == num:
            print("成功转化第"+str(i+1)+"个唐诗文件")

write_song_poet(254)

write_tang_poet(57)


file.close()