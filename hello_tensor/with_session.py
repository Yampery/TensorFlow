# -*- coding: utf-8 -*-
"""
Created on Fri May 11 08:54:58 2018

@author: Yampery
"""
""" 直接创建
import tensorflow as tf
a = tf.constant(3)  # 定义常量3
b = tf.constant(4)  # 定义常量4
# 建立session
with tf.Session() as sess:
    print('相加：%i' % sess.run(a+b))
    print('相乘：%i' % sess.run(a*b))
 """   
 
#  使用注入机制
import tensorflow as tf
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    print('相加：%i' % sess.run(add, feed_dict={a:3, b:4}))
    print('相乘：%i' % sess.run(mul, feed_dict={a:4, b:21}))
    print(sess.run([mul, add], feed_dict={a:3, b:4}))
    