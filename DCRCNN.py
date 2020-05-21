import keras
import tensorflow as tf

from keras.layers import Input, Subtract, Dense, Dropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K

A = K.constant([[0.35355, 0.49039, 0.46194, 0.41573, 0.35355, 0.27779, 0.19134, 0.09755],
                    [0.35355, 0.41573, 0.19134, -0.09755, -0.35355, -0.49039, -0.46194, -0.27779],
                    [0.35355, 0.27779, -0.19134, -0.49039, -0.35355, 0.09755, 0.46194, 0.41573],
                    [0.35355, 0.09755, -0.46194, -0.27779, 0.35355, 0.41573, -0.19134, -0.49039],
                    [0.35355, -0.09755, -0.46194, 0.27779, 0.35355, -0.41573, -0.19134, 0.49039],
                    [0.35355, -0.27779, -0.19134, 0.49039, -0.35355, -0.09755, 0.46194, -0.41573],
                    [0.35355, -0.41573, 0.19134, 0.09755, -0.35355, 0.49039, -0.46194, 0.27779],
                    [0.35355, -0.49039, 0.46194, -0.41573, 0.35355, -0.27779, 0.19134, -0.09755]],
                   dtype=tf.float32, shape=(8, 8))


class DCRCNN_1:
    def __init__(self, nb_filters, depth):
        self.nb_filters = nb_filters
        self.depth = depth

    def build(self):
        input_image = Input(shape=(None, None, 1), name='input_image')
        input_dc = Input(shape=(None, None, 1), name='input_dc')

        x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1], kernel_initializer='Orthogonal',
                   padding='same', name='conv_1')(input_image)
        x = Activation('relu', name='relu_1')(x)
        for i in range(self.depth - 2):
            x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1],
                       kernel_initializer='Orthogonal', use_bias=False, padding='same',
                       name='conv_{}'.format(i + 2))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn_{}'.format(i + 2))(x)
            x = Activation('relu', name='relu_{}'.format(i + 2))(x)

        x = Subtract(name='substract_1')([input_image, x])

        for j in range(3):
            x = Conv2D(filters=self.nb_filters * 2, kernel_size=[3, 3], strides=[2, 2],
                       kernel_initializer='Orthogonal', use_bias=False, padding='same',
                       name='conv_{}'.format(self.depth + j))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn_{}'.format(self.depth + j))(x)
            x = Activation('relu', name='relu_{}'.format(self.depth + j))(x)

        x = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1],
                   kernel_initializer='Orthogonal', use_bias=False, padding='same',
                   name='conv_{}'.format(self.depth + 4))(x)

        x = Subtract(name='substract_2')([input_dc, x])
        model = Model(inputs=[input_image, input_dc], outputs=x)

        return model


def DCT(x):
    def crop(dimension, h1, h2, w1, w2):
        # Crops (or slices) a Tensor on a given dimension from start to end
        # example : to crop tensor x[:, :, 5:10]
        # call slice(2, 5, 10) as you want to crop on the second dimension
        def func(x):
            if dimension == 0:
                return x[h1:h2]
            if dimension == 1:
                return x[:, h1:h2, w1:w2]
            if dimension == 2:
                return x[:, :, h1:h2, w1:w2]

        return Lambda(func)
    # output_list_1 = []
    # for k in range(768):
    #     output_list_2 = []
    #     for i in range(K.int_shape(x)[1] // 8):
    #         output_list_3 = []
    #         for j in range(K.int_shape(x)[2] // 8):
    #             block = K.slice(x, [k, i, j], [k + 1, 8, 8])
    #             # print(K.shape(K.dot(K.transpose(A), block)))
    #             # print(K.shape(K.dot(K.dot(K.transpose(A), block), A)))
    #             output_list_3.append(K.dot(K.dot(K.transpose(A), block), A)[:, 0, 0])
    #         output_list_2.append(K.stack(output_list_3, axis=1))
    #     output_list_1.append(K.stack(output_list_2, axis=0))
    # output = K.stack(output_list_1, axis=2)
    #
    # return output
    output_list_1 = []
    for i in range(K.int_shape(x)[1] // 8):
        output_list_2 = []
        for j in range(K.int_shape(x)[2] // 8):
            block = K.slice(x, [0, i, j], [200, 8, 8])
            block_dct = K.reshape(K.dot(K.dot(K.transpose(A), block), A), (200, 8, 8))[:, 0, 0]
            output_list_2.append(block_dct)
        output_list_1.append(K.stack(output_list_2, axis=1))
    output = K.stack(output_list_1, axis=2)

    return output


class DCRCNN_2:
    def __init__(self, input_shape, nb_filters, depth):
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.depth = depth

    def build(self):
        input_image = Input(shape=(self.input_shape[0], self.input_shape[1], 1), name='input_image')
        input_dc = Input(shape=(None, None, 1), name='input_dc')

        x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1], kernel_initializer='Orthogonal',
                   padding='same', name='conv_1')(input_image)
        x = Activation('relu', name='relu_1')(x)
        for i in range(self.depth - 2):
            x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1],
                       kernel_initializer='Orthogonal', use_bias=False, padding='same',
                       name='conv_{}'.format(i + 2))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn_{}'.format(i + 2))(x)
            x = Activation('relu', name='relu_{}'.format(i + 2))(x)
        x = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1],
                   kernel_initializer='Orthogonal', use_bias=False, padding='same',
                   name='conv_{}'.format(self.depth))(x)
        x = Subtract(name='substract')([input_image, x])
        x = Reshape((K.int_shape(x)[1], K.int_shape(x)[2]))(x)
        x = Lambda(DCT, output_shape=(4, 4))(x)

        model = Model(inputs=[input_image, input_dc], outputs=x)

        return model


"""
class DCRCNN_2:
    def __init__(self, nb_filters, depth):
        self.nb_filters = nb_filters
        self.depth = depth

    def build(self):
        input_image = Input(shape=(None, None, 1), name='input_image')
        input_dct = Input(shape=(None, None, 8, 8), name='input_dc')

        x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1], kernel_initializer='Orthogonal',
                   padding='same', name='conv_1')(input_image)
        x = Activation('relu', name='relu_1')(x)
        for i in range(self.depth - 2):
            x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1],
                       kernel_initializer='Orthogonal', use_bias=False, padding='same',
                       name='conv_{}'.format(i + 2))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn_{}'.format(i + 2))(x)
            x = Activation('relu', name='relu_{}'.format(i + 2))(x)

        for j in range(3):
            x = Conv2D(filters=self.nb_filters * 2, kernel_size=[3, 3], strides=[2, 2],
                       kernel_initializer='Orthogonal', use_bias=False, padding='same',
                       name='conv_{}'.format(self.depth + j))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn_{}'.format(self.depth + j))(x)
            x = Activation('relu', name='relu_{}'.format(self.depth + j))(x)

        x = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1],
                   kernel_initializer='Orthogonal', use_bias=False, padding='same',
                   name='conv_{}'.format(self.depth + 4))(x)
        y = K.zeros(shape=(8, 8, 1))
        x = Subtract(name='substract')([input_dct, x])
        model = Model(inputs=[input_image, input_dc], outputs=x)

        return model
"""