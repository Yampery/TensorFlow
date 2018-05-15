# -*- coding: utf-8 -*-
"""
Created on Tue May 15 08:28:18 2018
训练线性模型，添加保存检查点
模型：y = Wx + b
@author: Yampery
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义生成loss（损失值）可视化的函数
plotdata = { 'batchsize':[], 'loss':[] }

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return  [val if idx < w else sum(a[(idx-w):idx]) /w for idx, val in enumerate(a)]

# 生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

# 图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

tf.reset_default_graph()

# 创建模型
X = tf.placeholder('float')
Y = tf.placeholder('float')
# 模型参数
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
# 前向结构
z = tf.multiply(X, W) + b

# 反向优化 损失：生成值和真值之间的平方差
cost = tf.reduce_mean(tf.square(Y - z)) 
# 学习因子：调整参数的速度
learning_rate = 0.01
# 迭代调整：梯度下降法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 训练模型
# 初始化所有参数
init = tf.global_variables_initializer()
# 定义参数
training_epochs = 20
display_step = 2
# 创建一个saver，保存模型 max_to_keep=1 最多只保存一个检查点
saver = tf.train.Saver(max_to_keep=1)
savedir = 'log/'
# 启动Session
with tf.Session() as sess: 
    sess.run(init)
    # 向模型输入数据, 训练
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print('Epoch:', epoch+1, 'cost=', loss, 'W=', sess.run(W),
                  'b=', sess.run(b))
            if not (loss == 'NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
            # 保存训练模型
            saver.save(sess, savedir + 'linermodel.cpkt', global_step=epoch)
    # 训练结束
    print ('Finished!')
    print('cost=', sess.run(cost, feed_dict={X: train_X, Y: train_Y}),
          'W=', sess.run(W), 'b', sess.run(b))

    # 显示模型
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()
    
    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs Training loss')
    plt.show()

# 重启一个session，载入检查点
load_epoch = 18
with tf.Session() as sess_point:
    sess_point.run(tf.global_variables_initializer())
    saver.restore(sess_point, savedir + 'linermodel.cpkt-' + str(load_epoch))
    print('x=0.2, z=', sess_point.run(z, feed_dict={X :0.2}))







