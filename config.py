import os
import json
import math
import numpy as np
import tensorflow as tf

_mean_std = None
_img_shape = (224, 224)

def mean_std():
    global _mean_std
    if _mean_std == None:
        print("create new mean_std")
        with open('info/mean_std.txt', 'r', encoding='utf-8') as f:
            _mean_std = json.load(f)
    means = _mean_std['mean']
    stds = _mean_std['std']
    return means, stds

def img_shape():
    return _img_shape