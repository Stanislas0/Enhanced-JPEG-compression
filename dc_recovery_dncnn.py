import cv2
import os
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split

import DnCNN

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras_tqdm import TQDMCallback

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
    data_dir = "../image_dcfree"
    label_dir = "../image"
    data, label = [], []
    for i in range(1, 5001):
        file_name = os.path.join(data_dir, "{}.jpg".format(i))
        input_image = cv2.imread(file_name, 0)
        patches = patches_generator(input_image)
        # label_file_name = os.path.join(label_dir, "{}.jpg".format(i))
        # label_image = cv2.imread(label_file_name, 0)
        # label_image = label_image
        # label_patches = patches_generator(label_image)
        data.append(patches)
        # label.append(label_patches)
    data = np.array(data, dtype='uint8')
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    # label = np.array(label, dtype='int8')
    # label = label.reshape((label.shape[0] * label.shape[1], label.shape[2], label.shape[3], 1))

    np.save("../data_np/dcfree_image_crop.npy", data)
    # np.save("../data_np/original_image_crop.npy", label)
    print("Data generation finished.")


def dcfree_data_generator():
    dcfree_data = np.load("../data_np/image_dcfree_5000.npy")
    data = []
    for i in range(dcfree_data.shape[0]):
        input_image = dcfree_data[i].reshape((256, 256))
        patches = patches_generator(input_image)
        data.append(patches)

    data = np.array(data, dtype='int16')
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3], 1))
    np.save("../data_np/dcfree_image_crop.npy", data)
    print("Data generation finished.")


def train(model, input_data, label_data):
    learning_rate = 0.00001
    batch_size = 200
    nb_epoch = 100
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, label_data, test_size=0.2, shuffle=False)

    checkpoint_dir = "../checkpoints/dncnn_lfw_res/"
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
              validation_data=(X_test, Y_test), callbacks=[checkpointer,
                                                           TQDMCallback(leave_inner=True, leave_outer=True,
                                                                        metric_format="{name}: {value:0.6f}")])


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


def parallel_train(model, input_data, label_data):
    GPU_COUNT = 3
    learning_rate = 0.0001
    batch_size = 256
    nb_epoch = 100
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, label_data, test_size=0.2, shuffle=False)

    checkpoint_dir = "../checkpoints/dncnn_lfw_12_12_dcfree/"
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "checkpoint-epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5"),
        save_best_only=False, period=1)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    # model_dir = "../checkpoints/dncnn_lfw_8_8"
    # model_name = "checkpoint-epoch_70_val_loss_0.0011.hdf5"
    parallel_model = multi_gpu_model(model, gpus=GPU_COUNT)
    # parallel_model.load_weights(os.path.join(model_dir, model_name))
    parallel_model.compile(optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999),
                           loss='mean_squared_error', metrics=['accuracy'])
    parallel_model.fit(X_train, Y_train, batch_size=batch_size * GPU_COUNT, epochs=nb_epoch, verbose=1, shuffle=True,
                       validation_data=(X_test, Y_test), callbacks=[checkpointer])


def test(model, mode='single'):
    test_dir = "../image_rec/"
    # output_dir = "../results/DNCNN_v8_12_12_r_5000/"
    output_dir = "../image_out_2"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if mode == 'single':
        # model_dir = "../checkpoints/dncnn_lfw_res"
        # model_name = "checkpoint-epoch_100_val_loss_0.0012.hdf5"
        # model_dir = "../checkpoints/dncnn_lfw"
        # model_name = "checkpoint-epoch_100_val_loss_0.0013.hdf5"
        # model.load_weights(os.path.join(model_dir, model_name))
        model.load_weights("../Model/dncnn_lfw_12_12.h5")
        # model.save('my_model.h5')
        input_image = cv2.imread("../image_dcreplace/4019_out.jpg", 0) / 255.
        input_image = input_image.reshape((1, input_image.shape[0], input_image.shape[1], 1))
        output_image = model.predict(input_image) * 255.
        cv2.imwrite(os.path.join(output_dir, "4019_out.jpg"),
                    output_image.reshape((output_image.shape[1], output_image.shape[2])))
        #
        # for i in range(1, 5001):
        #     input_image = cv2.imread(os.path.join(test_dir, "{}_rec.jpg".format(i)), 0)
        #     input_image = input_image.reshape((1, input_image.shape[0], input_image.shape[1], 1)) / 255.0
        #     output_image = model.predict(input_image) * 255.0
        #     cv2.imwrite(os.path.join(output_dir, "{}_out.jpg".format(i)),
        #                 output_image.reshape((output_image.shape[1], output_image.shape[2])))

        print("Prediction done.")

    elif mode == 'multi':
        GPU_COUNT = 3
        model_dir = "../checkpoints/dncnn_lfw_12_12"
        model_name = "checkpoint-epoch_33_val_loss_0.0011.hdf5"
        parallel_model = multi_gpu_model(model, gpus=GPU_COUNT)
        parallel_model.load_weights(os.path.join(model_dir, model_name))
        model = parallel_model.layers[-2]
        model.save("./dncnn_lfw_12_12.h5")
        # for i in range(1, 501):
        #     input_image = cv2.imread(os.path.join(test_dir, "{}_rec.jpg".format(i)), 0)
        #     input_image = input_image.reshape((1, input_image.shape[0], input_image.shape[1], 1))
        #     output_image = model.predict(input_image)
        #     cv2.imwrite("../results/BSDS/{}_out.jpg".format(i),
        #                 output_image.reshape((output_image.shape[1], output_image.shape[2])))
        # print("Prediction done.")
    else:
        print("Please choose the right mode.")


def image_test(model):
    image = cv2.imread("../")
    model_dir = "../checkpoints/dncnn_lfw"
    model_name = "checkpoint-epoch_100_val_loss_0.0013.hdf5"
    # model_name = "single_model.h5"
    model.load_weights(os.path.join(model_dir, model_name))


def main():
    # Single GPU training
    # input_data = np.load("../data_np/recovered_image_crop_225000.npy").astype('float32')[0:112500] / 255.0
    # label_data = np.load("../data_np/original_image_crop_225000.npy").astype('float32')[0:112500] / 255.0
    # model = DnCNN.DnCNNRES(input_shape=(None, None, 1), nb_filters=64, depth=8)
    # model = model.build()
    # model.summary()
    # train(model, input_data, label_data)

    # Multi GPU training
    # input_data = np.load("../data_np/dcfree_image_crop_1125000.npy").astype('float32') / 255.0
    # label_data = np.load("../data_np/original_image_crop_1125000.npy").astype('float32') / 255.0
    # model = DnCNN.DnCNNRES(input_shape=(None, None, 1), nb_filters=64, depth=12)
    # model = model.build()
    # model.summary()
    # parallel_train(model, input_data, label_data)

    # Test
    model = DnCNN.DnCNNRES(input_shape=(None, None, 1), nb_filters=64, depth=12)
    # model = DnCNN.DnCNN(input_shape=(None, None, 1), nb_filters=64, depth=20)
    model = model.build()
    test(model, mode='single')
    # test(model, mode='multi')

    # data_generator()
    # dcfree_data_generator()


if __name__ == '__main__':
    main()
