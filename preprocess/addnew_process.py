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


if __name__ == "__main__":
    pass