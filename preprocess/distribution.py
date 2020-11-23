import os
import json
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def show_distribution(dir):
    label_cnt = {}

    with open('info/train/garbage_classify_rule.json', 'r', encoding='utf-8') as f:
        rule = json.load(f)
    
    with open(dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for img_path, label in data.items():
        if label in label_cnt:
            label_cnt[label] += 1
        else:
            label_cnt[label] = 1
    
    print(label_cnt)
    plt.bar(list(label_cnt.keys()), list(label_cnt.values()))
    plt.show()



if __name__ == "__main__":
    train_dir = 'info/train/train_data.txt'
    val_dir = 'info/validation/validation_data.txt'
    show_distribution(val_dir)
    pass