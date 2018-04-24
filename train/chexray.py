# -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
img_weight = 224
img_hight = 224
VOC_NUM_CLASS =14#batch_size
img_channels=3


def getfile(datapath,labelpath):
    class_train = []
    label_train = []
    # data = pd.read_csv(labelpath)
    #读取image 路径
    for root ,dirs,files in os.walk(datapath):
        for pic in files:

            class_train.append(os.path.join(root,pic))

    with open(labelpath) as filee:

        i = 0
        for l in filee.readlines():
            if i != 0:
                t = []

                for j in l.split(','):

                    t.append(j.strip('\n'))

                # t=np.array(t,dtype=int).tostring()
                label_train.append(t[0:14])

            else:

                i += 1
    temp=[]

    for i in label_train:
        array1=np.array(i,dtype=int).tostring()
        temp.append(array1)

    label_train=temp
    # label_train=str(label_train)

    return  class_train,label_train

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 2D tensor [batch_size], dtype=tf.float32
    '''

    image = tf.cast(image, tf.string)#把数据转换成tensorflow可以识别的数据
    label = tf.cast(label, tf.string)

    # # make an input queue ,并对一个数据集进行打乱
    input_queue = tf.train.slice_input_producer([image, label],shuffle=True)#生成一个输入队列

    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])#输出图片png格式的内容


    image = tf.image.decode_png(image_contents, channels=3)#读取png文件格式的内容进行读取


    ######################################
    # data argumentation should go to here
    ######################################
    image = tf.image.random_flip_left_right(image)

    # image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image.set_shape([image_H, image_W,3])


    # if you want to test the generated batches of images, you might want to comment the following line.

    # 如果想看到正常的图片，请注释掉111行（标准化）和 130行（image_batch = tf.cast(image_batch, tf.float32)）

    image = tf.image.per_image_standardization(image)#(减去均值除以方差对图像进行标准化)
    label = tf.decode_raw(label, tf.int64)#字符串类型转换为float32的向量

    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, [
        VOC_NUM_CLASS,
    ])
    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=64,
        capacity=capacity)

    #label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

def shulffedata(image,label):
    a = [int(i) for i in range(len(image))]
    random.shuffle(a)
    temp_image = []
    temp_label = []

    for i in a:
        temp_image.append(image[i])
        temp_label.append(label[i])

    image=temp_image
    label=temp_label

    return image,label


def dataprovider():
    train_dir = '/media/thomas/办公/images/'
    train_dircsv = '/media/thomas/办公/cxr8/Binarylabels.csv'
    # train_dir ='/home/thomas/文档/testimage/'
    # train_dircsv ='/home/thomas/文档/eBinarylabels.csv'

    BATCH_SIZE = 32

    images,labels = getfile(train_dir,train_dircsv)


    images,labels=shulffedata(images,labels)


    train_images=images[:(int(len(images)*0.8))]

    train_label=labels[:(int(len(images)*0.8))]
    test_images=images[int(len(images)*0.8):]
    test_label =labels[int(len(images)*0.8):]
    train_image_batch,train_label_batch = get_batch(train_images,train_label,img_weight,img_hight,BATCH_SIZE,100)
    test_image_batch,test_label_batch =get_batch(test_images,test_label,img_weight,img_hight,BATCH_SIZE,100)

    # with tf.Session() as sess:
    #     i = 0
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #     try:
    #         while not coord.should_stop() and i < 1:
    #             image, label = sess.run([test_image_batch, test_label_batch])
    #             # image, label = sess.run([train_image_batch, train_label_batch])
    #             # image,label = sess.run([image_batch,label_batch])
    #             for j in np.arange(BATCH_SIZE):
    #                 print("label:")
    #                 print(label[j])
    #                 plt.imshow(image[j, :, :, :])
    #                 plt.show()
    #             i += 1
    #     except tf.errors.OutOfRangeError:
    #         print("done!")
    #     finally:
    #
    #         coord.request_stop()
    #
    # coord.join(threads)

    return train_image_batch,train_label_batch,test_image_batch,test_label_batch




if __name__=='__main__':
    dataprovider()

















