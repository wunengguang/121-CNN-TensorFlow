from chexray import *
import numpy as np
from data_providers.chexnet import DenseNet

class_num=14
batch_size = 16
test_iteration=22420
growth_k = 32
init_learning_rate =0.01
dropout_rate =0.5



train_x ,train_y,test_x,test_y = dataprovider()

training_batch = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

label = tf.placeholder(tf.float32, shape=[None, VOC_NUM_CLASS])

trainingflag= tf.placeholder(tf.bool)
logits = DenseNet(x=training_batch,filters=growth_k,dropout_rate=dropout_rate,class_num=class_num,training=trainingflag).model

prediction =tf.nn.sigmoid(logits)

saver = tf.train.Saver()

#
with tf.Session() as sess:
    print('.....reading checkpoint....')
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('loading success !')
    else:
        print('loading fail!')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():

            for it in range(test_iteration):
                test_batch_x, test_batch_y = sess.run(
                    [test_x, test_y])

                test_feed_dict = {
                    training_batch: test_batch_x,
                    trainingflag:False
                }


                pred = sess.run([prediction],feed_dict=test_feed_dict)
                print(pred[0])
                print(test_batch_y)

                np.savetxt('./a.txt',pred[0])
                np.savetxt('./a.txt',test_batch_y)

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)