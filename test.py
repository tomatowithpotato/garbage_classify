import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import aug
from PIL import Image
import cv2

def test_aug():
    test_img_dir = r'C:\Users\admin\Desktop\临时\干电池.jpg'
    img = Image.open(test_img_dir)
    img = img.resize((224,224))
    img = np.array(img)
    img = np.asarray(img, np.float32) / 255.0
    
    img = aug.augumentor(img)
    for i in range(0,10):
        plt.imshow(img)
        plt.show()

def test_model():
    with open(r'info\train\garbage_classify_rule.json', 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    model = tf.saved_model.load(r'model\Alexnet.h5')
    test_img_dir = r'C:\Users\admin\Desktop\临时\干电池.jpg'
    image = tf.io.read_file(test_img_dir)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[224,224])
    image = tf.cast(image,tf.float32)
    image = image/255
    img_data = tf.expand_dims(image, axis=0)
    y_pred = np.argmax(model.__call__(img_data).numpy()) 
    print(label_dict[str(y_pred)])



def test_normalize():
    with open('info/image_label.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open('info/mean_std.txt', 'r', encoding='utf-8') as f:
        mean_std = json.load(f)
    
    means = mean_std['mean']
    stds = mean_std['std']

    for img_path in data.keys():
        tf_img = tf.io.read_file(img_path)
        tf_img = tf.image.decode_jpeg(tf_img,channels=3) 
        tf_img = np.asarray(tf_img, np.float32) / 255.0
        
        for i in range(3):
            tf_img[...,i] -= means[i]
            tf_img[...,i] /= stds[i]
        
        print(type(tf_img))
        plt.subplot(121)
        plt.imshow(tf_img)
        
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.asarray(img, np.float32) / 255.0
        print((224,224) + img.shape[2:])
        for i in range(3):
            img[...,i] -= means[i]
            img[...,i] /= stds[i]
        
        print(type(img))
        plt.subplot(122)
        plt.imshow(img)

        plt.show()


def test_resize(size=(224,224)):
    with open('info/image_label.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for img_path in data.keys():
        tf_img = tf.io.read_file(img_path)
        tf_img = tf.image.decode_jpeg(tf_img,channels=3)

        resize_scale = size[0] / max(tf_img.shape)
        new_shape = (int(tf_img.shape[0] * resize_scale), int(tf_img.shape[1] * resize_scale))
        print(tf_img.shape)
        print(new_shape)

        tf_img = tf.image.resize_with_pad(tf_img, 224, 224)
        tf_img = np.asarray(tf_img, np.float32) / 255.0

        plt.imshow(tf_img)
        plt.show()
        
        print('ok')

if __name__ == "__main__":
    test_resize()

