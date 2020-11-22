import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Dataset
import Dataset_gen
import Alexnet
import Resnet

if __name__ == "__main__": 
    epochs = 100 # 训练 epochs 
    batch_size = 32 # 批量大小 
    learning_rate = 1e-3 #学习率
    category = 40 #分类数目
    total = 19133

    model = Alexnet.Model(category)

    db_train, db_val = Dataset_gen.get_DatasetGenerator(batch_size=batch_size, num_classes=category)

    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        for batch_index, (X, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                y_pred = model(X)

                y_index = np.argmax(y, axis=1)
                y_pred_index = np.argmax(y_pred, axis=1)
                pred_max_present = np.argmax(np.bincount(y_pred_index))
                oh = np.argwhere(y_index == pred_max_present).size
                print(oh)

                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                loss = tf.reduce_mean(loss)
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
                bb = 22
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    '''

    
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