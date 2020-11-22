import os
import json
import numpy as np
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


def process(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[224,224])
    image = tf.cast(image,tf.float32)
    image = image/255
    return image, label

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
    dataset = dataset.map(process)
    if is_train is not None:
        dataset = dataset.shuffle(4000)
    dataset = dataset.batch(batch_size)

    print(dir + ' dataset OK')
    return dataset

def get_Dataset(num_classes, batch_size):
    train_dataset = get_dataset(num_classes, batch_size, 'info/train/train_data.txt', 'train')
    validation_dataset = get_dataset(num_classes, batch_size, 'info/validation/validation_data.txt')
    return train_dataset, validation_dataset

if __name__ == "__main__": 
    train_dataset = get_dataset(40, 64, 'info/train/train_data.txt')
    validation_dataset = get_dataset(40, 64, 'info/validation/validation_data.txt')
    
    print('Dataset.py')
    with open('info/train/garbage_classify_rule.json', 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    for image, label in validation_dataset:
        index = np.argmax(label[0])
        print(label_dict[str(index)])
        plt.imshow(image[0])
        plt.show()
    
    