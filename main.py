#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import matplotlib.pyplot as plt
import datetime
import shutil


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load vgg16 model and its pre-trained weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
# tests.test_load_vgg(load_vgg, tf)


def apply_conv2d_1x1(input_layer, num_classes, kernel_size=1):
    """
    Apply 1x1 2D Convolution to Tensor from VGG model (Encoder part of FCN)
    :param input_layer: Tensor from VGG model
    :param num_classes: Number of classes to classify
    :param kernel_size: Height and width of the 2D convolution window (this case is 1x1)
    :return: 1x1 2D Convoluted Output Tensor)
    """
    # tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ...,
    # kernel_initializer=None, ... , kernel_regularizer=None)
    return tf.layers.conv2d(input_layer, num_classes, kernel_size, padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


def apply_deconv2d(input_layer, num_classes, kernel_size=4, strides=(2, 2)):
    """
    Apply 2D deconvolution to Tensor from VGG model (Decoder part of FCN)
    :param input_layer: Tensor from VGG model
    :param num_classes: Number of classes to classify
    :param kernel_size: Height and width of the 2D deconvolution window (default: 4x4)
    :param strides: Strides of the deconvolution along the height and width (default: (2,2))
    :return: 2D Deconvoluted Output Tensor)
    """
    return tf.layers.conv2d_transpose(input_layer, num_classes, kernel_size, strides=strides, padding='same',
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    input_to_up_8 = apply_conv2d_1x1(vgg_layer7_out, num_classes)
    layer8 = apply_deconv2d(input_to_up_8, num_classes)

    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01)  # scaled to match vgg's original implementation
    input_to_skip_8 = apply_conv2d_1x1(pool4_out_scaled, num_classes)
    input_to_up_9 = tf.add(input_to_skip_8, layer8)  # skip connection
    layer9 = apply_deconv2d(input_to_up_9, num_classes)

    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001)  # scaled to match vgg's original implementation
    input_to_skip_9 = apply_conv2d_1x1(pool3_out_scaled, num_classes)
    input_to_up_10 = tf.add(input_to_skip_9, layer9)  # skip connection
    layer10 = apply_deconv2d(input_to_up_10, num_classes, kernel_size=16, strides=(8, 8))
    return layer10
# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # reshape output Tensor from 4D to 2D
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    labels = tf.reshape(correct_label, [-1, num_classes])

    # calculate cross entropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # add regularization term
    regularization_losses = tf.losses.get_regularization_losses()
    loss = tf.add_n([cross_entropy_loss] + regularization_losses)

    # Adam offers several advantages over the simple tf.train.GradientDescentOptimizer.
    # Foremost is that it uses moving averages of the parameters (momentum).
    # Simply put, this enables Adam to use a larger effective step size, and the algorithm will converge
    # to this step size without fine tuning. A simple tf.train.GradientDescentOptimizer could equally be used,
    # but would require more hyperparameter tuning before it would converge as quickly.
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return logits, train_op, loss
# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, model_dir):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param model_dir: directory to save trained model
    """
    # TODO: Implement function
    # initialize all the variables
    sess.run(tf.global_variables_initializer())

    # train model and evaluate loss
    total_loss = []
    for num_epoch in range(epochs):
        count = 0
        epoch_loss = 0
        for image, label in get_batches_fn(batch_size):
            feed_dict = {input_image: image,
                         correct_label: label,
                         keep_prob: 0.5,
                         learning_rate: 0.0005}

            _, batch_loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            count += 1
            epoch_loss += batch_loss
            print('Epoch: %i, Batch Loss: %.4f' % (num_epoch+1, batch_loss))

        epoch_loss = epoch_loss / count
        total_loss.append(epoch_loss)
        print('Epoch Loss: %.4f' % epoch_loss)
        print('********************************')

    # create directory to save model
    current_time = datetime.datetime.now()
    save_dir = os.path.join(model_dir, current_time.strftime('%Y-%m-%d'))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # save trained model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    model_path = os.path.join(save_dir, current_time.strftime('%H-%M-%S') + '.ckpt')
    saver.save(sess, model_path)

    # plot model loss
    x_plot = [i + 1 for i in range(epochs)]
    fig, ax = plt.subplots()
    ax.plot(x_plot, total_loss)
    ax.set(xlabel='Epoch', ylabel='Loss', title='Model Loss')
    ax.grid()

    # save plotted graph and visualize
    fig.savefig(os.path.join(save_dir, current_time.strftime('%H-%M-%S') + '.png'))
    plt.show()
# tests.test_train_nn(train_nn)


def restore_model(sess, model_dir, saved_date, saved_time):
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    restored_path = os.path.join(model_dir, saved_date)
    saver.restore(sess, os.path.join(restored_path, saved_time + '.ckpt'))


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    model_dir = './model'
    training_epochs = 40
    batch_size = 5
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        # https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, training_epochs, batch_size, get_batches_fn, train_op, loss, image_input,
                 correct_label, keep_prob, learning_rate, model_dir)

        # # restore trained model
        # restore_model(sess, model_dir, '2019-05-27', '22-11-20')

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
