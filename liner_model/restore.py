# -*- coding: utf-8 -*-
"""线性回归
"""

# 1. 准备数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return  [val if idx < w else sum(a[(idx-w):idx]) /w for idx, val in enumerate(a)]

train_x = np.linspace(-1, 1, 100)				
# 0.3添加噪声干扰
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3
plt.plot(train_x, train_y, 'ro', label='Original data')
plt.legend()
plt.show()

# 重置图
tf.reset_default_graph()

# 2. 创建模型(z = w * x + b)
# 2.1 正向搭建
# 占位符
X = tf.placeholder('float')
Y = tf.placeholder('float')
# 模型参数
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
# 前向结构
z = tf.multiply(X, W) + b

# 2.2 反向搭建模型（反向优化）
# 生成值和真值的平方差
cost = tf.reduce_mean(tf.square(Y - z)) 
# 调整参数的速度
learning_rate = 0.01
# 梯度下降法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 3. 迭代训练模型
# 3.1 训练模型
# 初始化所有的变量
init = tf.global_variables_initializer()
# 定义参数
# 迭代次数20
training_epochs = 20
display_step = 2
# 创建一个saver，保存模型
saver = tf.train.Saver()
savedir = 'log/'

    
# 载入模型
with tf.Session() as sess_load:
    sess_load.run(tf.global_variables_initializer())
    saver.restore(sess_load, savedir + 'linermodel.cpkt')
    print ('x=15, z=', sess_load.run(z, feed_dict={X: 15}))

