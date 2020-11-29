import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import aug
import config

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


def process(img_path, label):
    new_shape = config.img_shape()
    means, stds = config.mean_std()

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_with_pad(img, new_shape[0], new_shape[1])
    img = tf.cast(img, tf.float32) / 255.0

    tmp = [0, 0, 0]
    for i in range(3):
        tmp[i] = img[...,i] - means[i]
        tmp[i] = tmp[i] / stds[i]
    img = tf.stack(tmp, 2)

    #img = tf.numpy_function(aug.augumentor, [img], img.dtype)

    return img, label


def train_process(img_path, label):
    img, label = process(img_path, label)
    img_shape = tf.shape(img)
    img = tf.numpy_function(aug.augumentor, [img], img.dtype)
    img = tf.reshape(img, img_shape)
    return img, label


def get_dataset(numclasses, batch_size, dir, is_train=None):
    image_paths = []
    labels = []

    with open(dir, 'r', encoding='utf-8') as f:
        image_label = json.load(f)
    for image_path, label in image_label.items():
        image_paths.append(image_path)
        labels.append(label)
    
    labels = tf.keras.utils.to_categorical(labels, numclasses)
    # 标签平滑
    labels = smooth_labels(labels)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if is_train is not None:
        #训练
        dataset = dataset.map(train_process)
        dataset = dataset.shuffle(2000)
    else:
        #测试
        dataset = dataset.map(process)

    dataset = dataset.batch(batch_size)

    print(dir + ' dataset OK')
    return dataset


def get_Dataset(num_classes, batch_size):
    print('Raw Dataset')
    train_dataset = get_dataset(num_classes, batch_size, 'info/train/train_data.txt', 'train')
    validation_dataset = get_dataset(num_classes, batch_size, 'info/validation/validation_data.txt')
    return train_dataset, validation_dataset


if __name__ == "__main__": 
    train_dataset, validation_dataset = get_Dataset(batch_size=32, num_classes=40)
    
    print('Dataset.py')
    with open('info/train/garbage_classify_rule.json', 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    for image, label in validation_dataset:
        index = np.argmax(label[0])
        print(label_dict[str(index)])
        #a=image[0].numpy()
        plt.imshow(image[0])
        plt.show()
    
    