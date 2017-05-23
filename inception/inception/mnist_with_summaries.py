"""Test inception/slim to train mnist dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
# import numpy as np
import argparse
import sys
import tensorflow as tf

# import inception.slim as slim
from inception.slim.scopes import arg_scope
from inception.slim.ops import conv2d, max_pool, fc, flatten, dropout
from inception.slim.losses import cross_entropy_loss

# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None


# TODO
# Application logic will be added here
def model(features):
    """Model function for coarse stack."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    tf.summary.image('input', input_layer, 10)
    # print("input_layer: ", input_layer.get_shape())

    with arg_scope([conv2d, fc], stddev=0.1, bias=0.1,
                   weight_decay=0.0005, activation=tf.nn.relu):
        # See https://github.com/tensorflow/models/blob/master/inception/
        # inception/slim/ops.py for more details.
        #   net = conv2d(inputs=[batch_size, height, width, channels],
        #       num_filters_out, kernel_size=[kernel_height, kernel_width],
        #       stride=int|[stride_height, stride_width],
        #       padding='VALID'|'SAME',
        #       scope='conv1')
        #   net = max_pool(inputs,
        #                           kernel_size=[kernel_height, kernel_width],
        #                           stride=int|[stride_height, stride_width],
        #                           scope='pool1')

        net = conv2d(input_layer, 32, kernel_size=[5, 5],
                     stride=1, padding='SAME', scope='conv1')
        print("conv1: ", net.get_shape())
        net = max_pool(net, kernel_size=[2, 2], stride=[2, 2],
                       padding='SAME', scope='pool1')
        print("pool1: ", net.get_shape())
        net = conv2d(net, 64, kernel_size=[5, 5], stride=1,
                     padding='SAME', scope='conv2')
        print("conv2: ", net.get_shape())
        net = max_pool(net, kernel_size=[2, 2], stride=[2, 2],
                       padding='SAME', scope='pool2')
        print("pool2: ", net.get_shape())

        net = flatten(net, scope='flatten3')
        print("flatten3: ", net.get_shape())
        net = fc(net, num_units_out=1024, activation=None, scope='fc3')
        print("fc3: ", net.get_shape())
        net = dropout(net, keep_prob=FLAGS.dropout, scope='dropout3')
        print("dropout3: ", net.get_shape())

        net = fc(net, num_units_out=10, activation=None, scope='readout')
        print("readout: ", net.get_shape())
        # net = dropout(net, 0.5, scope='dropout7')
        # net = fc(net, 1000, activation=None, scope='fc8')

        tf.summary.tensor_summary('inception_net', net)

        return net

def train():
    # Get input data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Placeholders
    # x is a 2d tensor which will hold N x 784 (i.e. flattened 28x28 mnist images)
    # y_ will hold the one-hot encoded integer value
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # images = tf.Variable(tf.random_normal([1, 240, 320, 3], stddev=0.35),
    #                      name="images")
    # depths = tf.Variable(tf.random_normal([1, 55, 74], stddev=0.35),
    #                      name="depths")
    # model = coarse_stack(images, depths)

    y_conv = model(x)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, save the
    # variables to disk.
    with tf.Session() as sess:

        # Do some work with the model.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Merge all summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        # Add an op to initialize the variables.
        sess.run(tf.global_variables_initializer())

        for i in range(FLAGS.max_steps):
            batch = mnist.train.next_batch(50)
            if i % 10 == 0: # Record summaries and test-set accuracy
                # train_accuracy = accuracy.eval(feed_dict={
                #     x: batch[0], y_: batch[1]
                #     })
                summary, acc = sess.run([merged, accuracy], feed_dict={
                    x: mnist.test.images, y_: mnist.test.labels
                })
                test_writer.add_summary(summary, i)
                print("Accurary at step %s: %s" %(i, acc))
            else: # Record train set summaries and train
                if i % 100 == 99: # Record execution stats
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                        options = run_options,
                        run_metadata = run_metadata,
                        feed_dict={
                            x: batch[0], y_: batch[1]
                            }
                        )
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                else: # Record a summary
                    summary, _ = sess.run([merged, train_step],
                        feed_dict={
                            x: batch[0], y_: batch[1]
                            }
                        )
                    train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()

            # train_step.run(feed_dict={
            #     x: batch[0], y_: batch[1]
            #     })

        # print("test accuracy %g" %accuracy.eval(feed_dict={
        #     x: mnist.test.images, y_: mnist.test.labels
        # }))

        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)

def main(unused_argv):
    """Main."""
    # TODO
    print("In main...")
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == "__main__":
    print("Before run...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    # tf.app.run()
