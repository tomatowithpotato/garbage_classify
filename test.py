import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == "__main__":
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


