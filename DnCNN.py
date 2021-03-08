from keras.models import Model
from keras.layers import Input, Subtract, Dense, Dropout, Activation, Flatten, Add
from keras.layers import Conv2D, MaxPooling2D, Deconvolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization


class DnCNN:
    def __init__(self, input_shape, nb_filters, depth):
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.depth = depth

    def build(self):
        # model = Sequential()
        # model.add(Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1], kernel_initializer='Orthogonal',
        #                  input_shape=self.input_shape, padding='same', name='conv_1'))
        # model.add(Activation('relu', name='relu_1'))
        # for i in range(self.depth - 2):
        #     model.add(Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1],
        #                      kernel_initializer='Orthogonal', use_bias=False, padding='same',
        #                      name='conv_{}'.format(i + 2)))
        #     model.add(BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn_{}'.format(i + 2)))
        #     model.add(Activation('relu', name='relu_{}'.format(i + 2)))
        # model.add(Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1],
        #                  kernel_initializer='Orthogonal', use_bias=False, padding='same',
        #                  name='conv_{}'.format(self.depth)))
        # model.add(Flatten())
        # model.add(Dense(1, activation='relu'))

        input_image = Input(shape=self.input_shape, name='input')
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
        model = Model(inputs=input_image, outputs=x)

        return model


class DnCNNRES:
    def __init__(self, input_shape, nb_filters, depth):
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.depth = depth

    def build(self):
        input_image = Input(shape=self.input_shape, name='input')
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

        # x = Subtract(name='substract')([input_image, x])
        x = Add(name='add_1')([input_image, x])
        x_shortcut = x
        x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1], kernel_initializer='Orthogonal',
                   padding='same', name='conv_{}'.format(self.depth + 1))(x)
        x = Activation('relu', name='relu_{}'.format(self.depth + 1))(x)
        for i in range(self.depth - 2):
            x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1],
                       kernel_initializer='Orthogonal', use_bias=False, padding='same',
                       name='conv_{}'.format(i + self.depth + 2))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn_{}'.format(i + self.depth + 2))(x)
            x = Activation('relu', name='relu_{}'.format(i + self.depth + 2))(x)
        x = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1],
                   kernel_initializer='Orthogonal', use_bias=False, padding='same',
                   name='conv_{}'.format(self.depth + self.depth))(x)

        # x = Subtract(name='substract_2')([x_shortcut, x])
        x = Add(name='add_2')([x_shortcut, x])
        model = Model(inputs=input_image, outputs=x)

        return model


class DnCNNRES2:
    def __init__(self, input_shape, nb_filters, depth):
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.depth = depth

    def build(self):
        input_image = Input(shape=self.input_shape, name='input')
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

        # x = Subtract(name='substract')([input_image, x])
        x = Add(name='add_1')([input_image, x])
        x = Activation('relu', name='relu_s_1')(x)
        x_shortcut = x
        x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1], kernel_initializer='Orthogonal',
                   padding='same', name='conv_{}'.format(self.depth + 1))(x)
        x = Activation('relu', name='relu_{}'.format(self.depth + 1))(x)
        for i in range(self.depth - 2):
            x = Conv2D(filters=self.nb_filters, kernel_size=[3, 3], strides=[1, 1],
                       kernel_initializer='Orthogonal', use_bias=False, padding='same',
                       name='conv_{}'.format(i + self.depth + 2))(x)
            x = BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001, name='bn_{}'.format(i + self.depth + 2))(x)
            x = Activation('relu', name='relu_{}'.format(i + self.depth + 2))(x)
        x = Conv2D(filters=1, kernel_size=[3, 3], strides=[1, 1],
                   kernel_initializer='Orthogonal', use_bias=False, padding='same',
                   name='conv_{}'.format(self.depth + self.depth))(x)

        # x = Subtract(name='substract_2')([x_shortcut, x])
        x = Add(name='add_2')([x_shortcut, x])
        x = Activation('relu', name='relu_s_2')(x)
        model = Model(inputs=input_image, outputs=x)

        return model