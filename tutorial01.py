from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


def lenet(images, keep_prob):
    """
    Args:
        images: a 4-D tensor that holds batched input images
    Return:
        A tensor that contains classification probabilities result, and a dictionary
        of all intermediate tensors.
    """    
    end_points = {}
    # Input shape of 28,28,1 and -1 is just for TF purposes
    end_points['images'] = tf.reshape(images, [-1, 28, 28, 1])
    
    # Define the scope
    with tf.variable_scope('conv1'):
        # Define the weights for the convolution aka. Kernel size, kernel size, stride and number of filters
        w1 = tf.get_variable('weights', [5, 5, 1, 32])
        
        # Define the bias
        b1 = tf.get_variable('biases', [32],
                             initializer=tf.zeros_initializer())
        # Perform the computation and apply ReLU function
        # First do the conv2d with weights w1 in the SAME namespace then add the bias, later activation function
        end_points['conv1'] = tf.nn.relu(
            tf.nn.conv2d(end_points['images'], w1, [1, 1, 1, 1], 'SAME') + b1)
    
    # Add a max-pooling operation with kernel 2x2 and stride 1
    end_points['pool1'] = tf.nn.max_pool(
        end_points['conv1'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    
    with tf.variable_scope('conv2'):
        w2 = tf.get_variable('weights', [5, 5, 32, 64])
        b2 = tf.get_variable('biases', [64],
                             initializer=tf.zeros_initializer())
        end_points['conv2'] = tf.nn.relu(
            tf.nn.conv2d(end_points['pool1'], w2, [1, 1, 1, 1], 'SAME') + b2)
    end_points['pool2'] = tf.nn.max_pool(
        end_points['conv2'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    
    end_points['flatten'] = tf.reshape(end_points['pool2'], [-1, 7 * 7 * 64])
    with tf.variable_scope('fc3'):
        w3 = tf.get_variable('weights', [7 * 7 * 64, 1024])
        b3 = tf.get_variable('biases', [1024],
                             initializer=tf.zeros_initializer())
        end_points['fc3'] = tf.nn.relu(tf.matmul(end_points['flatten'], w3) + b3)
        
    end_points['dropout'] = tf.nn.dropout(end_points['fc3'], keep_prob)
    with tf.variable_scope('fc4'):
        w4 = tf.get_variable('weights', [1024, 10])
        b4 = tf.get_variable('biases', [10],
                             initializer=tf.zeros_initializer())
        end_points['fc4'] = tf.matmul(end_points['fc3'], w4) + b4
    
    return end_points['fc4'], end_points
# NOTE: You don't need to run this code snippet since we have already trained it
# and it will consume lots of resources on our server.

mnist = input_data.read_data_sets("C:/Users/Zhou/Documents/GitHub/CrTutorial/Intel/Tutorial01/data/MNIST_data", one_hot=True)

g = tf.Graph()
with g.as_default():
    images = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    logits, end_points = lenet(images, keep_prob)
    
    # Nodes for training
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    train = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(25000):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            _, loss_val = sess.run([train, loss],
                                   feed_dict={images: batch_xs,
                                              labels: batch_ys,
                                              keep_prob: 0.5})
        
            if i % 100 == 0:
                print('Loss value of a training batch at step %5d: %f' % (i, np.mean(loss_val)))
            if i % 1000 == 0:
                acc = sess.run(accuracy,
                               feed_dict={images: mnist.test.images,
                                          labels: mnist.test.labels,
                                          keep_prob: 1.0})
                print('Accuracy after running %5d steps: %f' % (i, acc))
        
        # save the trained model
        saver.save(sess, "./mnist_lenet_log/tutorial01.ckpt")