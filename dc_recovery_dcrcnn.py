import cv2
import os
import glob
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split

import DCRCNN

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)


def patches_generator(image):
    [h, w] = image.shape
    patches = []
    patch_size = 32
    stride = 16
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = image[i:i + patch_size, j:j + patch_size]
            patches.append(x)

    return patches


def data_generator():
    data_dir = "../image_rec"
    label_dir = "../image"
    data, label = [], []
    for i in range(1, 5001):
        file_name = os.path.join(data_dir, "{}_rec.jpg".format(i))
        input_image = cv2.imread(file_name, 0)
        patches = patches_generator(input_image)
        label_file_name = os.path.join(label_dir, "{}.jpg".format(i))
        label_image = cv2.imread(label_file_name, 0)
        label_image = label_image
        label_patches = patches_generator(label_image)
        data.append(patches)
        label.append(label_patches)
    data = np.array(data, dtype='uint8')
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    label = np.array(label, dtype='uint8')
    label = label.reshape((label.shape[0] * label.shape[1], label.shape[2], label.shape[3], 1))

    np.save("../data_np/recovered_image_crop_300000.npy", data)
    np.save("../data_np/original_image_crop_300000.npy", label)
    print("Data generation finished.")


def dct_generator():
    # data = np.load("../data_np/recovered_dct_data_5000.npy")
    label = np.load("../data_np/dct_data_5000.npy")
    patch_size = 4
    stride = 2
    h, w = 32, 32
    dct, label_dct = [], []
    for k in range(label.shape[0]):
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                # dct.append(data[k, i:i + patch_size, j:j + patch_size])
                label_dct.append(label[k, i:i + patch_size, j:j + patch_size])

    # np.save("../data_np/recovered_dct_data_crop.npy", dct)
    np.save("../data_np/dct_data_crop.npy", label_dct)
    print("Data generation finished.")


def train(model, input_data, label_data):
    learning_rate = 0.0001
    batch_size = 128
    nb_epoch = 100
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, label_data, test_size=0.2, shuffle=False)

    checkpoint_dir = "../checkpoints/dncnn_BSDS_res/"
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "checkpoint-epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5"),
        save_best_only=False, period=1)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True,
              validation_data=(X_test, Y_test), callbacks=[checkpointer, lr_scheduler])


def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch <= 10:
        lr = initial_lr
    elif epoch <= 20:
        lr = initial_lr / 10
    elif epoch <= 30:
        lr = initial_lr / 20
    elif epoch <= 40:
        lr = initial_lr / 30
    else:
        lr = initial_lr / 40

    return lr


def parallel_train(model, input_data, dc_data, label_data):
    GPU_COUNT = 3
    learning_rate = 0.01
    batch_size = 256
    nb_epoch = 100
    index = int(input_data.shape[0] * 0.8)
    # X_train, X_test, Y_train, Y_test = train_test_split(input_data, label_data, test_size=0.2, shuffle=False)
    X_train, X_test = input_data[0:index], input_data[index:]
    Y_train, Y_test = label_data[0:index], label_data[index:]
    dc_train, dc_test = dc_data[0:index], dc_data[index:]
    checkpoint_dir = "../checkpoints/DCRCNN_1"
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "checkpoint-epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5"),
        save_best_only=False, period=1)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    parallel_model = multi_gpu_model(model, gpus=GPU_COUNT)
    parallel_model.compile(optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999),
                           loss='mean_squared_error', metrics=['accuracy'])
    parallel_model.fit([X_train, dc_train], Y_train, batch_size=batch_size * GPU_COUNT, epochs=nb_epoch, verbose=1,
                       shuffle=True, validation_data=([X_test, dc_test], Y_test),
                       callbacks=[checkpointer, lr_scheduler])


def test(model, mode='single'):
    test_dir = "../image_rec/"
    output_dir = "../results/DCRCNN/"
    dct = []
    if mode == 'single':
        test_dct = np.load("../data_np/recovered_dct_data_5000.npy")[:, :, :, 0, 0]
        test_dct = test_dct.reshape((-1, 1, 32, 32, 1))
        for i in range(1, 501):
            input_image = cv2.imread(os.path.join(test_dir, "{}_rec.jpg".format(i)), 0)
            input_image = input_image.reshape((1, input_image.shape[0], input_image.shape[1], 1)) / 255.0
            dct.append(model.predict([input_image, test_dct[i]]))
            # output_image = model.predict(input_image) * 255.0
            # cv2.imwrite(os.path.join(output_dir, "{}_out.jpg".format(i)),
            # output_image.reshape((output_image.shape[1], output_image.shape[2])))
        np.save("../data_np/DCRCNN_pred_dc.npy", dct)
        print("Prediction done.")
    elif mode == 'multi':
        GPU_COUNT = 3
        model_dir = "../checkpoints/DCRCNN"
        model_name = "checkpoint-epoch_74_val_loss_21.2462.hdf5"
        parallel_model = multi_gpu_model(model, gpus=GPU_COUNT)
        parallel_model.load_weights(os.path.join(model_dir, model_name))
        model = parallel_model.layers[-2]
        model.save("./DCRCNN.h5")
        # for i in range(1, 501):
        #     input_image = cv2.imread(os.path.join(test_dir, "{}_rec.jpg".format(i)), 0)
        #     input_image = input_image.reshape((1, input_image.shape[0], input_image.shape[1], 1))
        #     output_image = model.predict(input_image)
        #     cv2.imwrite("../results/BSDS/{}_out.jpg".format(i),
        #                 output_image.reshape((output_image.shape[1], output_image.shape[2])))
        # print("Prediction done.")
    else:
        print("Please choose the right mode.")


def main():
    # Data generation
    # data_generator()
    # dct_generator()

    # Single GPU training
    # input_data = np.load("../data_np/BSDS_input_data_40.npy").astype('float32') / 255.0
    # label_data = np.load("../data_np/BSDS_label_data_40_res.npy").astype('float32') / 255.0
    # model = DnCNN.DnCNN(input_shape=(None, None, 1), nb_filters=64, depth=17)
    # model = model.build()
    # model.summary()
    # train(model, input_data, label_data)

    # Multi GPU training
    # input_data = np.load("../data_np/recovered_image_crop_225000.npy").astype('float32') / 255.0
    # dc_data = np.load("../data_np/recovered_dct_data_crop_225000.npy")[:, :, :, 0, 0].reshape((-1, 4, 4, 1))
    # label_data = np.load("../data_np/dct_data_crop_225000.npy")[:, :, :, 0, 0].reshape((-1, 4, 4, 1))
    # model = DCRCNN.DCRCNN_1(nb_filters=64, depth=17)
    # model = model.build()
    # model.summary()
    # parallel_train(model, input_data, dc_data, label_data)

    # Test
    model = DCRCNN.DCRCNN_1(nb_filters=64, depth=17)
    model = model.build()
    test(model, mode='single')
    # test(model, mode='multi')


if __name__ == '__main__':
    main()
