import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def create_image_label(save_file, img_ROOT, labels_ROOT):
    image_label = {}
    
    for filename in os.listdir(labels_ROOT):
        with open(os.path.join(labels_ROOT, filename), 'r') as f:
            lst = f.read().split(', ')
            if int(lst[1]) > 39:
                print('error')
                return None
            image_label[os.path.join(img_ROOT, lst[0])] = int(lst[1])
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(image_label, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    ROOT = 'E:/学习资料/神圣罗马帝国'
    img_ROOT = ROOT + '/garbage_data/train_data/image'
    labels_ROOT = ROOT + '/garbage_data/train_data/label'
    save_file = 'info/image_label.txt'
    create_image_label(save_file, img_ROOT, labels_ROOT)
