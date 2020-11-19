import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Dataset


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #使用CPU

class ResBlock(tf.keras.layers.Layer):
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
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        identity=self.downsample(input)
        output=tf.keras.layers.add([out,identity])
        output=tf.nn.relu(output)
        return output


class ResNet(tf.keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        # 预处理层
        self.stem=tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,(3,3),strides=(1,1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
        ])
        # resblock
        self.layer1=self.build_resblock(64,layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1],stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # there are [b,512,h,w]
        # 自适应
        self.avgpool=tf.keras.layers.GlobalAveragePooling2D()
        self.fc=tf.keras.layers.Dense(num_classes)

    def call(self, input, training=None):
        x=self.stem(input)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
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

def resnet18(category):
    return  ResNet([2,2,2,2], category)


if __name__ == "__main__": 
    epochs = 100 # 训练 epochs 
    batch_size = 16 # 批量大小 
    learning_rate = 1e-3 #学习率
    category = 40 #分类数目
    total = 19133

    model = resnet18(category)

    db_train, db_val = Dataset.get_Dataset(category, batch_size)

    # 装配     
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = "categorical_crossentropy", metrics=['accuracy']) 

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience = 5,min_delta=1e-3)
    ] 

    # 训练和验证 
    model.fit(db_train, epochs=epochs, validation_data=db_val)

    #保存模型
    tf.saved_model.save(model, r'model\Resnet.h5')