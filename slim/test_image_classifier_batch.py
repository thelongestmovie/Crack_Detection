from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import math
import time
import numpy as np
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
from manager import GPUManager

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'test_dir', '', 'Test image directory.')

tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')

tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_resnet_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'cuda_visible_devices', '1', 'device to test')
FLAGS = tf.app.flags.FLAGS


def ListFile(jpg_dir):
    jpg_list = []
    print (jpg_dir + 'jpg_data')
    for jpg_file in os.listdir(jpg_dir + 'jpg_data'):
        if '.jpg' in jpg_file:
            jpg_list.append(jpg_file)
    return jpg_list

def Write_txt(image_id,predictions):
    for i in range(idx, idx_end):
        print('{} {}'.format(image_id, predictions[i - idx].tolist()))

def main(_):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_visible_devices
    gm = GPUManager()
    with gm.auto_choice():
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default() as g:
            ####################
            # Select the model #
            ####################
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
                is_training=False)
            #####################################
            # Select the preprocessing function #
            #####################################
            preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name,
                is_training=False)
            test_image_size = FLAGS.test_image_size or network_fn.default_image_size
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path

            batch_size = FLAGS.batch_size
            # tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
            tensor_input = tf.placeholder(tf.float32, [batch_size, test_image_size, test_image_size, 3])
            logits, _ = network_fn(tensor_input)
            #logits = tf.nn.top_k(logits, 5)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            print (FLAGS.test_dir)
            jpg_list = ListFile(FLAGS.test_dir)
            tot = len(jpg_list)
            with tf.Session(config=config,graph = g) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_path)
                time_start = time.time()
                with open(os.path.join(FLAGS.test_dir, 'test.txt'), 'w') as f:
                    for idx in range(0, tot, batch_size):
                        images = list()
                        idx_end = min(tot, idx + batch_size)
                        print(idx)
                        for i in range(idx, idx_end):
                            image_id = jpg_list[i]
                            test_path = os.path.join(FLAGS.test_dir, 'jpg_data',image_id)
                            image = open(test_path, 'rb').read()
                            image = tf.image.decode_jpeg(image, channels=3)
                            processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
                            processed_image = sess.run(processed_image)
                            images.append(processed_image)
                        images = np.array(images)
                        predictions = sess.run(logits, feed_dict={tensor_input: images})
                        for i in range(idx, idx_end):
                            # if predictions[i-idx][0]-predictions[i-idx][1] <= 0.8:
                            #     continue
                            f.write('{}:{}\n'.format(jpg_list[i], np.argmax(predictions[i - idx])))
                            #print(np.argmax(predictions[i - idx]))
                            #print('{}:{}\n'.format(jpg_list[i], sess.run(tf.argmax(predictions[i - idx]))))
                            #print('{} {}'.format(jpg_list[i], predictions[i - idx]))

                time_total = time.time() - time_start
                print('total time: {}, total images: {}, average time: {}'.format(
                    time_total, tot, time_total / tot))
if __name__ == '__main__':
    tf.app.run()