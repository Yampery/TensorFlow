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
# 启动Session
with tf.Session() as sess: 
    sess.run(init)
    # 存放批次值和损失值
    plotdata = { 'batchsize':[], 'loss':[] }
    # 向模型输入数据, 训练
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_x, Y:train_y})
            print('Epoch:', epoch+1, 'cost=', loss, 'W=', sess.run(W),
                  'b=', sess.run(b))
            if not (loss == 'NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)
    # 训练结束
    print ('Finished!')
    # 保存训练模型
    saver.save(sess, savedir + 'linermodel.cpkt')
    print('cost=', sess.run(cost, feed_dict={X: train_x, Y: train_y}),
          'W=', sess.run(W), 'b', sess.run(b))

    # print ('x = 0.2, z = ', sess.run(z, feed_dict={X: 0.2}))
            
# 3.2 训练模型可视化
    # 图形显示
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fittedline')
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
    # 使用模型
    print ('x = 0.2, z = ', sess.run(z, feed_dict={X: 0.2}))