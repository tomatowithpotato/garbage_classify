import os
import json
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def sk_split(test_size=0.2):
    dir = 'info/image_label.txt'
    
    with open(dir, 'r', encoding='utf-8') as f:
        image_label = json.load(f)
    
    imgs = list(image_label.keys())
    labels = list(image_label.values())
    print(len(imgs),len(labels))
    print(type(imgs[0]), type(labels[0]))
    print(imgs[0], labels[0])

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(imgs, labels, test_size=test_size, random_state=0)
    
    print(len(train_imgs))
    print(len(val_imgs))
    print(len(train_imgs)+len(val_imgs))

    train_data = dict(zip(train_imgs, train_labels))
    val_data = dict(zip(val_imgs, val_labels))

    print(len(train_data))
    print(len(val_data))
    print(len(train_data)+len(val_data))
    print("ok")
    return train_data, val_data

def save_in_file(train_data, val_data):
    print(len(train_data) + len(val_data))
    with open('info/train/train_data.txt', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open('info/validation/validation_data.txt', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4, ensure_ascii=False)
    print('good man!!!')


if __name__ == "__main__":
    #create_train_validation()
    train_data, val_data = sk_split()
    save_in_file(train_data, val_data)
    pass
