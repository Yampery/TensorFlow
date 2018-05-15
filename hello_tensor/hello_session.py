# -*- coding: utf-8 -*-
"""
Created on Fri May 11 08:49:05 2018

@author: Yampery
"""
import tensorflow as tf
hello = tf.constant('Hello, Tensorflow!')
sess = tf.Session()
print (sess.run(hello))
sess.close()

