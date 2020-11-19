import os
import json
import numpy as np
import Data_generator
import tensorflow as tf
import matplotlib.pyplot as plt

# 标签平滑
def smooth_labels(y, smooth_factor=0.1):
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def get_dataset_gen(num_classes, batch_size, dir, is_train=False):
    image_paths = []
    labels = []
    with open(dir, 'r', encoding='utf-8') as f:
        image_label = json.load(f)
    for image_path, label in image_label.items():
        image_paths.append(image_path)
        labels.append(label)
    
    labels = tf.keras.utils.to_categorical(labels, num_classes)

    dataset = Data_generator.BaseSequence(image_paths, labels, batch_size, [224, 224], is_train)

    print(dir + ' dataset OK')
    return dataset

def get_DatasetGenerator(batch_size, num_classes):
    train_dataset = get_dataset_gen(batch_size, num_classes, 'info/train/train_data.txt', True)
    validation_dataset = get_dataset_gen(batch_size, num_classes, 'info/validation/validation_data.txt')
    return train_dataset, validation_dataset

if __name__ == "__main__":
    train_dataset, validation_dataset = get_DatasetGenerator(num_classes=40, batch_size=64)
    
    with open('info/train/garbage_classify_rule.json', 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    for image, label in validation_dataset:
        #print(label_dict[str(label[0])])
        print(label_dict[str(np.argmax(label[0]))])
        plt.imshow(image[0])
        plt.show()