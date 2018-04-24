# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
from tflearn.layers.conv import global_avg_pool
from chexray import *
import time
import shutil
from datetime import timedelta
from models.chexnet import DenseNet

# 模型参数设置
growth_k = 32
init_learning_rate =0.01
dropout_rate =0.5
batch_size = 32
iteration = 2803
total_epochs =20
class_num=14
image_size=224
weight_decay = 1e-4

#-- 验证数据集 - -----------------------
#
# def Evaluate(sess):
#     test_acc = 0.0
#     test_loss = 0.0
#
#     for it in range(test_iteration):
#         test_batch_x, test_batch_y = sess.run(
#             [test_x, test_y])
#
#         test_feed_dict = {
#             training_batch: test_batch_x,
#             label: test_batch_y,
#             learning_rate: epoch_learning_rate,
#             trainingflag: False
#         }
#         batch_loss, batch_acc, pred, testlabel = sess.run(
#             [cost, acc, prediction, label],
#             feed_dict=test_feed_dict)
#         test_loss += batch_loss
#         test_acc += batch_acc
#
#     mean_test_accuray = test_acc / test_iteration
#     mean_test_loss = test_loss / test_iteration
#     test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
#                                      tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
#
#     return mean_test_accuray, mean_test_loss, test_summary

# -----------------------评估准确率------------------------------
def Evaluate_acc(final_tensor, label):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.round(final_tensor), tf.round(label))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 所有label的平均准确率
        tf.summary.scalar('accuracy', accuracy)
    return accuracy





training_batch = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])#占位符

label = tf.placeholder(tf.float32, shape=[None, VOC_NUM_CLASS])#占位符

trainingflag= tf.placeholder(tf.bool)#占位符

learning_rate = tf.placeholder(tf.float32, name='learning_rate')#占位符

#-------------------interfer-------------------

logits = DenseNet(x=training_batch,filters=growth_k,dropout_rate=dropout_rate,class_num=class_num,training=trainingflag).model#加载模型训练

# logits = ChexNet(inputdata=training_batch,growth_rate=growth_rate,n_class=14, is_training=trainingflag, keep_prob=keep_prob)

prediction = tf.nn.sigmoid(logits)#出现概率

cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits,targets=label,pos_weight=10))#损失

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])#对损失加上l2来防止过拟合。

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#使用adamoptimizer

train = optimizer.minimize(cost+ l2_loss*weight_decay)#最小化损失

# acc = Evaluate_acc(logits, label)#真确率评估
acc = Evaluate_acc(prediction, label)
saver = tf.train.Saver(tf.global_variables())

train_x ,train_y,test_x,test_y = dataprovider()#获取数据

#---------建立会话----------------------------------

with tf.Session() as sess:
    # tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)#保存日志文件
    epoch_learning_rate = init_learning_rate#初始化学习率

    print('start training......')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    s_time =time.time()
    try:
        while not coord.should_stop():
            for epoch in range(1, total_epochs + 1):
                #每一个epcoh之后学习率缩小
                if epoch !=1:
                    epoch_learning_rate = epoch_learning_rate / 10
                start_time = time.time()#得到初始时间
                total_loss = 0.0
                total_accuracy = 0.0
                for step in range(1, iteration + 1):
                    image_batchs, label_batchs = sess.run(
                        [train_x , train_y])#获取训练集

                    train_feed_dict = {
                        training_batch: image_batchs,
                        label: label_batchs,
                        learning_rate: epoch_learning_rate,
                        trainingflag: True
                    }
                    #进行训练，
                    _, batch_loss, batch_acc, pre, x = sess.run(
                        [train, cost, acc, prediction, label],
                        feed_dict=train_feed_dict)
                    train_loss = batch_loss#
                    train_acc=batch_acc
                    #获取训练的信息
                    train_summary = tf.Summary(
                            value=[tf.Summary.Value(tag='train_mean_loss', simple_value=train_loss),
                                   tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                    # 对训练信息进行保存
                    summary_writer.add_summary(summary=train_summary, global_step= total_epochs)
                    summary_writer.flush()

                    line = "epoch: %d, train_loss: %.4f, train_acc: %.4f \n" % (
                            epoch, train_loss, train_acc )#打印出训练的准确率
                    print(line)

                    # --------保存日志日志文件-------
                    with open('./logs/logs.txt', 'a') as f:
                            f.write(line)

                time_per_epoch = time.time() - start_time  # 一轮时间
                seconds_left = int((total_epochs - epoch) * time_per_epoch)  # 剩余时间
                print("每一轮时间: %s, 剩余完成时间: %s" % (
                        str(timedelta(seconds=time_per_epoch)),
                        str(timedelta(seconds=seconds_left))))  # 打印出一时间和剩余的时间
                saver.save(sess=sess, save_path='./trainmodel/dense.ckpt', global_step=epoch)#每一轮训练完成后对模型进行保存
            total_training_time = time.time() - s_time  # 总共时间
            print("\nTotal training time: %s" % str(timedelta(
                        seconds=total_training_time)))

    except tf.errors.OutOfRangeError:
        print('训练第%d,eopch'%epoch)
    finally:
        coord.request_stop()
    coord.join(threads)















