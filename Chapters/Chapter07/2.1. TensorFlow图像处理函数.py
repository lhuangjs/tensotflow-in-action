# -*- coding: utf-8 -*-
"""
Created on Sat May  5 17:02:22 2018

@author: huangjs
"""

import tensorflow as tf   
import numpy as np

with tf.Session() as sess:
    directory = "../../datasets/csv_test/*.*"
    file_names=tf.train.match_filenames_once(directory)
    print(sess.run(file_names))