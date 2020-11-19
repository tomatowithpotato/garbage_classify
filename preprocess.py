import os
import json
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def rename():
    cnt = 166
    ROOT = r'E:\学习资料\神圣罗马帝国\垃圾数据集\data_baidu\可回收物_金属厨具'
    for file_name in os.listdir(ROOT):
        os.rename(os.path.join(ROOT, file_name), os.path.join(ROOT, 'img_bd_{}.jpg'.format(cnt)))
        cnt += 1


def create_label():
    ROOT = r'E:\学习资料\神圣罗马帝国\垃圾数据集\data_baidu\可回收物_金属厨具'
    for file_name in os.listdir(ROOT):
        dir = os.path.join(ROOT, file_name)
        pre = dir.split('.')[0]
        suf = dir.split('.')[1]
        if suf == 'jpg':
            label_dir = pre + '.txt'
            with open(label_dir, 'w') as f:
                f.write(file_name + ', ' + str(40))

def show_distribution():
    with open(r'info\train\garbage_classify_rule.json', 'r', encoding='utf-8') as f:
        rule = json.load(f)
    label_cnt = {}
    ROOT = r'E:\学习资料\神圣罗马帝国\垃圾数据集\train_data'
    for file_name in os.listdir(ROOT):
        dir = os.path.join(ROOT, file_name)
        suf = dir.split('.')[1]
        if suf == 'txt':
            with open(dir, 'r') as f:
                label = rule[f.read().split(', ')[1]]
            if label in label_cnt:
                label_cnt[label] += 1
            else:
                label_cnt[label] = 1
    print(label_cnt)



if __name__ == "__main__":
    pass