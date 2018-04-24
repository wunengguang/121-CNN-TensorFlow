# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten
from tflearn.layers.conv import global_avg_pool
import numpy as np

def conv_layer(input,filter,kernel,stride=1,layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input,use_bias=False,filters=filter,kernel_size=kernel,strides=stride,padding="SAME")
        return network

#全局平均采样
def Global_Average_Pooling(x):
    return global_avg_pool(x,name='Global_avg_pooling')
#防止过拟合
def Drop_out(x,rate,training):
    return tf.layers.dropout(inputs=x,rate=rate,training=training)
#激活函数
def Relu(x):
    return tf.nn.relu(x)
#平均池化
def Average_pooling(x,pool_size=[2,2],stride=2,padding='VALID'):
    return  tf.layers.average_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)
#最大池化
def Max_pooling(x,pool_size=[3,3],stride=2,padding='SAME'):
    return tf.layers.max_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

#连接
def Concatenation(layers):
    return tf.concat(layers,axis=3)

def weight_variable_msra(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.contrib.layers.variance_scaling_initializer())

def weight_variable_xavier( shape, name):
    return tf.get_variable(
        name,
        shape=shape,
        initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)




#定义稠密卷积神经网络，
class DenseNet():
    def __init__(self,x,filters,dropout_rate,class_num,training):
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.filters = filters
        self.training = training
        self.model=self.Dense_net(x)

    def new_conv_layer(self, bottom, filter_shape, name):
        with tf.variable_scope(name)as scope:
            w = tf.get_variable("w", shape=filter_shape, initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                "b",
                shape=filter_shape[-1],
                initializer=tf.constant_initializer(0.))
            conv = tf.nn.conv2d(bottom, w, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)

        return bias

    def batch_norm(self, x,training):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = [0]

        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        output = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return output

    # def batch_norm(self, x, training,):
    #     output = tf.contrib.layers.batch_norm(
    #         x, scale=True, is_training=training,
    #         updates_collections=None)
    #     return output

    def transition_layer_to_classes(self,_input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output =self.batch_norm(_input, training=self.training)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        output = Global_Average_Pooling(output)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = weight_variable_xavier(
            [features_total,self.class_num], name='W')
        bias = bias_variable([self.class_num])
        logits = tf.matmul(output, W) + bias
        return logits

    def new_transition_layer_to_class(self,_input,input_size, output_size, name):
        # BN
        output = self.batch_norm(_input, training=self.training)
        # ReLU
        output = tf.nn.relu(output)

        # average pooling
        output = Global_Average_Pooling(output)

        shape = output.get_shape().to_list()
        dim = np.prod(shape[1:])
        x = tf.reshape(output, [-1, dim])

        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                "W",
                shape=[input_size, output_size],
                initializer=tf.random_normal_initializer(0., 0.01))
            b = tf.get_variable(
                "b",
                shape=[output_size],
                initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add(tf.matmul(x, w), b, name=scope)

        return fc



    #定义bottleneck_layer
    def bottleneck_layer(self,x,scope):
        with tf.name_scope(scope):
            x = self.batch_norm(x, training=self.training)
            x = Relu(x)
            x = conv_layer(x,filter=4*self.filters,kernel=[1,1],layer_name=scope+'_conv1')
            x = Drop_out(x,rate=self.dropout_rate,training=self.training)
            x = self.batch_norm(x, training=self.training)
            x = Relu(x)
            x = conv_layer(x,filter=self.filters,kernel=[3,3],layer_name=scope+'_conv2' )
            x = Drop_out(x,rate=self.dropout_rate,training=self.training)

            return x
    #定义转换层
    def trasition_layer(self,x,scope):
        with tf.name_scope(scope):
            filtere = int(int(x.get_shape()[-1]))
            x = self.batch_norm(x, training=self.training)

            x = Relu(x)

            x = conv_layer(x,filter=filtere,kernel=[1,1],layer_name=scope+'_conv1')
            x = Drop_out(x,rate=self.dropout_rate,training=self.training)
            x = Average_pooling(x,pool_size=[2,2],stride=2)
            return x
    #定义密集连接模块
    def dense_block(self,input_x,nb_layers,layer_name):

        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x,scope=layer_name+'_bottleN_'+str(0))
            layers_concat.append(x)
            for i in range(nb_layers-1):
                # if i!=0:
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x,scope=layer_name + '_bottleN_'+str(i+1))
                layers_concat.append(x)
            x = Concatenation(layers_concat)
            return x
    #密集连接网络
    def Dense_net(self,input_x):
        x = conv_layer(input_x,filter=2*self.filters,kernel=[7,7],stride=2,layer_name='conv0')
        x = Max_pooling(x)
        x = self.dense_block(input_x=x,nb_layers=6,layer_name='dense_1')
        x = self.trasition_layer(x,scope='trans_1')
        x = self.dense_block(input_x=x,nb_layers=12,layer_name='dense_2')
        x = self.trasition_layer(x,scope='trans_2')
        x = self.dense_block(input_x=x,nb_layers=24,layer_name='dense_3')
        x = self.trasition_layer(x,scope='trans_3')
        x = self.dense_block(input_x=x,nb_layers=16,layer_name='dense_4')

        logits = self.transition_layer_to_classes(x)


        return logits





