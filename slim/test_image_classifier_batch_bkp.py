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
import cv2
# from PIL import Image

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
    img_path = []
    img_name = []
    for jpg_file in os.listdir(os.path.join(jpg_dir , 'jpg_data')):
        if '.jpg' in jpg_file:
            img_path.append(os.path.join(jpg_dir,'jpg_data',jpg_file))
            img_name.append(jpg_file)
    return img_path,img_name

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
        # with tf.Graph().as_default():
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
            tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
            logits, _ = network_fn(tensor_input)
            #logits = tf.nn.top_k(logits, 5)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            img_path, img_name = ListFile(FLAGS.test_dir)
            img_paths = tf.convert_to_tensor(img_path, tf.string)
            img_names = tf.convert_to_tensor(img_name, tf.string)
            img_path, img_name = tf.train.slice_input_producer([img_paths, img_names],  num_epochs = 1,shuffle=False)
            image = tf.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
            image_batch, name_batch = tf.train.batch([processed_image,img_name], batch_size=batch_size, num_threads=32, capacity=1024,
                                                  allow_smaller_final_batch=False)


            with open(os.path.join(FLAGS.test_dir, 'test.txt'), 'w') as f:
                with tf.Session(config = config,graph = g) as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    saver = tf.train.Saver()
                    saver.restore(sess, checkpoint_path)
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess, coord)

                    idx = 0
                    time_data,time_infer,time_write = 0,0,0
                    try:
                        while not coord.should_stop():
                            time1 = time.time()
                            print ('************')
                            image_batch_v, name_batch_v = sess.run([image_batch, name_batch])
                            time_data = time_data + time.time()-time1
                            time2 = time.time()
                            predictions = sess.run(logits, feed_dict={tensor_input: image_batch_v})
                            time_infer = time_infer + time.time() - time2
                            time3 = time.time()
                            # for i in range(0,len(predictions)):
                            #     if(predictions[i][0] > predictions[i][1]):
                            #         print (name_batch_v[i],predictions[i])
                            # print (len(name_batch_v))
                            for i in range(0, batch_size):
                                # if predictions[i][0]-predictions[i][1] <= 0.8:
                                #     continue
                                idx += 1
                                f.write('{}:{}\n'.format(name_batch_v[i], np.argmax(predictions[i])))
                            time_write = time_write + time.time() - time3
                    except tf.errors.OutOfRangeError:
                        print ('kill all threads!!!!!!')
                    finally:
                        coord.request_stop()
                        print('all threads are asked to stop!')
                    coord.join(threads)
                    print('all threads are stopped!')
                    print (idx)
                    print(time_data,time_infer,time_write)


            # with open(os.path.join(FLAGS.test_dir, 'test.txt'), 'w') as f:
            #     with tf.Session(config=config,graph = g) as sess:
            #         sess.run(tf.global_variables_initializer())
            #         saver = tf.train.Saver()
            #         saver.restore(sess, checkpoint_path)
            #         time_start = time.time()
            #         for idx in range(0, tot, batch_size):
            #             batch_begin = time.time()
            #             images = list()
            #             idx_end = min(tot, idx + batch_size)
            #             print(idx)
            #             img_read = 0
            #             img_process = 0
            #             img_process1 = 0
            #             for i in range(idx, idx_end):
            #                 img_begin = time.time()
            #                 image_id = jpg_list[i]
            #                 test_path = os.path.join(FLAGS.test_dir, 'jpg_data',image_id)
            #                 img_read += time.time()-img_begin
            #                 process_begin = time.time()
            #                 # image = cv2.imread(test_path)
            #                 with tf.gfile.FastGFile(test_path, 'r') as f1:
            #                     image = f1.read()
            #                 image = tf.image.decode_jpeg(image, channels=3)
            #                 img_process1 += time.time() - process_begin
            #                 processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
            #                 processed_image = sess.run(processed_image)
            #                 # processed_image = cv2.resize(image, (test_image_size, test_image_size), interpolation=cv2.INTER_LINEAR)
            #                 img_process += time.time() - process_begin
            #                 images.append(processed_image)
            #
            #             print('Read images:{},images process:{},images process1:{}'.format(img_read,img_process,img_process1))
            #             infer_begin = time.time()
            #
            #             images = np.array(images)
            #             predictions = sess.run(logits, feed_dict={tensor_input: images})
            #
            #             print('Infer time:{}'.format(time.time()-infer_begin))
            #             write_begin = time.time()
            #
            #             for i in range(idx, idx_end):
            #                 # print(predictions[i-idx][0],predictions[i-idx][1])
            #                 # if abs(predictions[i-idx][0]-predictions[i-idx][1]) <= 0.00000005:
            #                 #     continue
            #                 f.write('{}:{}\n'.format(jpg_list[i], np.argmax(predictions[i - idx])))
            #
            #             print('Write time:{}'.format(time.time()-write_begin))
            #             print ('One batch spend:{}'.format(time.time()-batch_begin))
            #
            #
            #         time_total = time.time() - time_start
            #         print('total time: {}, total images: {}, average time: {}'.format(
            #             time_total, tot, time_total / tot))
if __name__ == '__main__':
    tf.app.run()