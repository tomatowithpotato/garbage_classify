import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Dataset
import Dataset_aug
import Alexnet
import Resnet

if __name__ == "__main__": 
    epochs = 100 # 训练 epochs 
    batch_size = 32 # 批量大小 
    learning_rate = 1e-3 #学习率
    category = 40 #分类数目
    total = 19133

    model = Alexnet.Model(category)

    db_train, db_val = Dataset_aug.get_DatasetGenerator(batch_size=batch_size, num_classes=category)

    
    # 装配     
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = "categorical_crossentropy", metrics=['accuracy']) 

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience = 5,min_delta=1e-3)
    ] 

    # 训练和验证 
    model.fit(db_train, epochs=epochs, validation_data=db_val)

    #保存模型
    tf.saved_model.save(model, 'model/Alexnet.h5')

    #11:06:54.404295
    #18:46:52.348865
