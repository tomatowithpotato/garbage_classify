import os
import json
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def sample(all_data, splt_rate):
    all_size = len(all_data)
    train_data = []
    validation_data = []
    interval = math.ceil(1/splt_rate)
    
    splits = [x for x in range(0, all_size, interval)]
    splits.append(all_size)
    
    for index in range(len(splits)-1):
        split = all_data[splits[index]:splits[index+1]]
        validation_element_index = random.randint(0, min(len(split),interval)-1)
        validation_element = split.pop(validation_element_index)
        validation_data.append(validation_element)
        train_data.extend(split)
    
    return train_data, validation_data


def create_train_validation():
    dir = 'info/image_label.txt'
    
    with open(dir, 'r', encoding='utf-8') as f:
        image_label = json.load(f)

    #验证集占总数据的20%
    splt_rate = 0.2

    trainset = {}
    validationset = {}

    categorys = {}
    for key, value in image_label.items():
        if value in categorys:
            categorys[value].append(key)
        else:
            categorys[value] = [key]
    
    print(len(categorys))

    t1 = {}
    t2 = {}
    for key, value in categorys.items():
        t1[key], t2[key] = sample(value, splt_rate)
    
    for key, value in t1.items():
        for v in value:
            trainset[v] = key
    for key, value in t2.items():
        for v in value:
            validationset[v] = key
    
    print(len(trainset) + len(validationset))
    with open('info/train/train_data.txt', 'w', encoding='utf-8') as f:
        json.dump(trainset, f, indent=4, ensure_ascii=False)
    with open('info/validation/validation_data.txt', 'w', encoding='utf-8') as f:
        json.dump(validationset, f, indent=4, ensure_ascii=False)
    print('good man!!!')


if __name__ == "__main__":
    create_train_validation()
    pass