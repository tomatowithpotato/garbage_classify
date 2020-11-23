import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Dataset

@DeprecationWarning
class ResBlock(tf.keras.layers.Layer):
    """
    二联残差块
    """
    def __init__(self, filter_num, stride=1):
        super(ResBlock, self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filter_num,(3,3),strides=stride,padding='same')
        self.bn1=tf.keras.layers.BatchNormalization()
        self.relu=tf.keras.layers.Activation('relu')

        self.conv2=tf.keras.layers.Conv2D(filter_num,(3,3),strides=1,padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride!=1:
            self.downsample=tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filter_num,(1,1),strides=stride))
        else:
            self.downsample=lambda x:x

    def call(self, input, training=None):
        out=self.conv1(input)
        out=self.bn1(out, training=training)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out, training=training)

        identity=self.downsample(input)
        output=tf.keras.layers.add([out,identity])
        output=tf.nn.relu(output)
        return output

@DeprecationWarning
class Model(tf.keras.Model):
    """
    二联残差网络
    """
    def __init__(self, num_classes=10, layer_dims=[2,2,2,2]):
        super(Model, self).__init__()
        self.save_path = 'model/Resnet.h5'
        # 预处理层
        self.stem=tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,(3,3),strides=(1,1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
        ])
        # resblock
        self.layer1 = self.build_resblock(64,layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1],stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # there are [b,512,h,w]
        # 自适应
        self.avgpool=tf.keras.layers.GlobalAveragePooling2D()
        self.fc=tf.keras.layers.Dense(num_classes)

    def call(self, input, training=None):
        x=self.stem(input, training=training)
        x=self.layer1(x, training=training)
        x=self.layer2(x, training=training)
        x=self.layer3(x, training=training)
        x=self.layer4(x, training=training)
        # [b,c]
        x=self.avgpool(x)
        x=self.fc(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks= tf.keras.Sequential()
        # may down sample
        res_blocks.add(ResBlock(filter_num,stride))
        # just down sample one time
        for pre in range(1,blocks):
            res_blocks.add(ResBlock(filter_num,stride=1))
        return res_blocks