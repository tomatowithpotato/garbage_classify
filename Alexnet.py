import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Dataset
import Dataset_gen


class Model(tf.keras.Model):
    def __init__(self, category):
        super().__init__()   
        self.conv1 = tf.keras.layers.Conv2D(
            filters=96,             # 卷积层神经元（卷积核）数目
            kernel_size=[11, 11],   # 感受野大小
            strides=[4,4],          # 步长
            padding='valid',         # padding策略（vaild 或 same）
            activation=tf.nn.relu   # 激活函数
        )

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2,2], padding='valid')

        self.conv2 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[5, 5],
            strides=[1,1],
            padding='valid',
            activation=tf.nn.relu
        )

        self.bn2 = tf.keras.layers.BatchNormalization()

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2,2], padding='valid')

        self.conv3 = tf.keras.layers.Conv2D(
            filters=384,
            kernel_size=[3, 3],
            strides=[1,1],
            padding='valid',
            activation=tf.nn.relu
        )

        self.conv4 = tf.keras.layers.Conv2D(
            filters=384,
            kernel_size=[3, 3],
            strides=[1,1],
            padding='valid',
            activation=tf.nn.relu
        )

        self.conv5 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            strides=[1,1],
            padding='valid',
            activation=tf.nn.relu
        )

        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2,2], padding='valid')

        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=category)

    #@tf.function
    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 128, 128, 32]
        x = self.bn1(x)
        x = self.pool1(x)                       # [batch_size, 64, 64, 32]
        x = self.conv2(x)                       # [batch_size, 64, 64, 64]
        x = self.bn2(x)
        x = self.pool2(x)                       # [batch_size, 32, 32, 64]
        x = self.conv3(x)                       # [batch_size, 32, 32, 128]
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)                       # [batch_size, 16, 16, 128]
        x = self.flatten(x)                     # [batch_size, 16 * 16 * 128]
        x = self.dense1(x)                      # [batch_size, 512]
        x = self.dense2(x)                      # [batch_size, 128]
        x = self.dense3(x)                      # [batch_size, 4] 因为有4种情况
        output = tf.nn.softmax(x)
        return output