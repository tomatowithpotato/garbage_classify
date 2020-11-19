import os
import math
import codecs
import random
import numpy as np
from glob import glob
from PIL import Image
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from aug import augumentor


class BaseSequence(tf.keras.utils.Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels, batch_size, img_size, train=False):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def center_img(img, size=None, fill_value=255):
        """
        center img in a square background
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def img_aug(self, img):
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
        dic_parameter = {'flip_horizontal': random.choice([True, False]),
                         'flip_vertical': random.choice([True, False]),
                         'theta': random.choice([0, 0, 0, 90, 180, 270])
                        }


        img_aug = data_gen.apply_transform(img, transform_parameters=dic_parameter)
        return img_aug

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = Image.open(img_path)
        resize_scale = self.img_size[0] / max(img.size[:2])
        img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.convert('RGB')
        img = np.array(img)

        # 数据归一化
        img = np.asarray(img, np.float32) / 255.0
        
        mean = [0.56719673, 0.5293289, 0.48351972]
        std = [0.20874391, 0.21455203, 0.22451781]
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        img[..., 0] /= std[0]
        img[..., 1] /= std[1]
        img[..., 2] /= std[2]


        # 数据增强
        if self.train:
            # img = self.img_aug(img)
            img = augumentor(img)
        img = self.center_img(img, self.img_size[0])
        return img


        ########################################
        # img = Image.open(img_path)
        # img = img.resize((self.img_size[0], self.img_size[0]))
        # img = img.convert('RGB')
        # img = np.array(img)
        # img = img.astype(np.float)
        # # if self.train:
        # #     # img = self.img_aug(img)
        # #     img = augumentor(img)
        # img = img[:, :, ::-1]
        #
        # return img
        ########################################
        # Img = Image.open(img_path)
        # Img = cv2.cvtColor(np.asarray(Img), cv2.COLOR_RGB2BGR)
        # Img = cv2.resize(Img, (self.img_size[0], self.img_size[0]))
        # Img = Img[:, :, (2, 1, 0)]
        # Img = np.asarray(Img)
        # Img = Img.astype(np.float)
        # return Img




    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)