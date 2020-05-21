"""
DC recovery from DC-free JPEG images.
Author: Qinkai ZHENG

Package version:
cv2 3.4.2
numpy 1.15.4
"""

import cv2
import numpy as np


# Initialisation
patch_size = 8
image_shape = (256, 256)
[w, h] = image_shape
w_n = w // patch_size
h_n = h // patch_size
DC_range = np.arange(-64, 65)

# Quantization matrix
Q = [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 130, 99]]


def read_image(image_path):
    """
    Read an image and convert it in gray scale.
    :param image_path: string, path of image file
    :return: image: ndarray, gray scale image
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (w, h, ))
    return image


def dct_transform(image, mode='compress', dc_free=False):
    """
    Calculate DCT coefficients of an image.
    :param image: ndarray, input image
    :param mode: string, compression or normal mode
    :param dc_free: boolean, option whether DC coefficient is missing
    :return: dct_coefs
    """
    dct_coefs = np.zeros([w_n, h_n, patch_size, patch_size])
    image = image - 128 * np.ones([w, h])
    for i in range(w_n):
        for j in range(h_n):
            image_patch = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            dct = cv2.dct(image_patch)
            if dc_free:
                dct[0][0] = 0.
            if mode == 'compress':
                dct_coefs[i, j] = np.round(dct / Q)
            elif mode == 'normal':
                dct_coefs[i, j] = dct

    return dct_coefs


def estimate_0(dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    mse_min = np.inf
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    for dc in DC_range:
        mse1 = 0.
        mse2 = 0.
        mse3 = 0.
        dct_target[0, 0] = dc
        spatial_target = cv2.idct(dct_target * Q) + 128
        if dct_left is not False:
            mse1 += np.sum(np.square(spatial_left[0:8, 7] - spatial_target[0:8, 0])) / 8
            mse2 += np.sum(np.square(spatial_left[0:7, 7] - spatial_target[1:8, 0])) / 7
            mse3 += np.sum(np.square(spatial_left[1:8, 7] - spatial_target[0:7, 0])) / 7
        if dct_up is not False:
            mse1 += np.sum(np.square(spatial_up[7, 0:8] - spatial_target[0, 0:8])) / 8
            mse2 += np.sum(np.square(spatial_up[7, 0:7] - spatial_target[0, 1:8])) / 7
            mse3 += np.sum(np.square(spatial_up[7, 1:8] - spatial_target[0, 0:7])) / 7
        if dct_right is not False:
            mse1 += np.sum(np.square(spatial_right[0:8, 0] - spatial_target[0:8, 7])) / 8
            mse2 += np.sum(np.square(spatial_right[0:7, 0] - spatial_target[1:8, 7])) / 7
            mse3 += np.sum(np.square(spatial_right[1:8, 0] - spatial_target[0:7, 7])) / 7
        if dct_down is not False:
            mse1 += np.sum(np.square(spatial_down[0, 0:8] - spatial_target[7, 0:8])) / 8
            mse2 += np.sum(np.square(spatial_down[0, 0:7] - spatial_target[7, 1:8])) / 7
            mse3 += np.sum(np.square(spatial_down[0, 1:8] - spatial_target[7, 0:7])) / 7
        mse = np.min([mse1, mse2, mse3])
        if mse < mse_min:
            mse_min = mse
            dc_optimal = dc

    return dc_optimal


def estimate(dct_target, dct_left=False, dct_up=False, dct_right=False, dct_down=False):
    """
    Estimation of DCT coefficients with the help of adjacent blocks.
    :param dct_target: ndarray, DCT of target block
    :param dct_left: ndarray, DCT of left block
    :param dct_up: ndarray, DCT of upper block
    :param dct_right: ndarray, DCT of right block
    :param dct_down: ndarray, DCT of nether block
    :return: dc_optimal: float, optimal value of DC prediction
    """
    mse_min = np.inf
    dc_optimal = 0.
    if dct_left is not False:
        spatial_left = cv2.idct(dct_left * Q) + 128
    if dct_up is not False:
        spatial_up = cv2.idct(dct_up * Q) + 128
    if dct_right is not False:
        spatial_right = cv2.idct(dct_right * Q) + 128
    if dct_down is not False:
        spatial_down = cv2.idct(dct_down * Q) + 128
    for dc in DC_range:
        mse1 = 0.
        mse2 = 0.
        mse3 = 0.
        dct_target[0, 0] = dc
        spatial_target = cv2.idct(dct_target * Q) + 128
        if dct_left is not False:
            mse1 += np.sum(np.square((spatial_left[0:8, 7] - spatial_target[0:8, 0]) -
                                     (spatial_left[0:8, 6] - spatial_left[0:8, 7]))) / 8
            mse2 += np.sum(np.square((spatial_left[0:7, 7] - spatial_target[1:8, 0]) -
                                     (spatial_left[0:7, 6] - spatial_left[1:8, 7]))) / 7
            mse3 += np.sum(np.square((spatial_left[1:8, 7] - spatial_target[0:7, 0]) -
                                     (spatial_left[1:8, 6] - spatial_left[0:7, 7]))) / 7
        if dct_up is not False:
            mse1 += np.sum(np.square((spatial_up[7, 0:8] - spatial_target[0, 0:8]) -
                                     (spatial_up[6, 0:8] - spatial_up[7, 0:8]))) / 8
            mse2 += np.sum(np.square((spatial_up[7, 0:7] - spatial_target[0, 1:8]) -
                                     (spatial_up[6, 0:7] - spatial_up[7, 1:8]))) / 7
            mse3 += np.sum(np.square((spatial_up[7, 1:8] - spatial_target[0, 0:7]) -
                                     (spatial_up[6, 1:8] - spatial_up[7, 0:7]))) / 7
        if dct_right is not False:
            mse1 += np.sum(np.square((spatial_right[0:8, 0] - spatial_target[0:8, 7]) -
                                     (spatial_right[0:8, 1] - spatial_right[0:8, 0]))) / 8
            mse2 += np.sum(np.square((spatial_right[0:7, 0] - spatial_target[1:8, 7]) -
                                     (spatial_right[0:7, 1] - spatial_right[1:8, 0]))) / 7
            mse3 += np.sum(np.square((spatial_right[1:8, 0] - spatial_target[0:7, 7]) -
                                     (spatial_right[1:8, 1] - spatial_right[0:7, 0]))) / 7
        if dct_down is not False:
            mse1 += np.sum(np.square((spatial_down[0, 0:8] - spatial_target[7, 0:8]) -
                                     (spatial_down[1, 0:8] - spatial_down[0, 0:8]))) / 8
            mse2 += np.sum(np.square((spatial_down[0, 0:7] - spatial_target[7, 1:8]) -
                                     (spatial_down[1, 0:7] - spatial_down[0, 1:8]))) / 7
            mse3 += np.sum(np.square((spatial_down[0, 1:8] - spatial_target[7, 0:7]) -
                                     (spatial_down[1, 1:8] - spatial_down[0, 0:7]))) / 7
        mse = np.min([mse1, mse2, mse3])
        if mse < mse_min:
            mse_min = mse
            dc_optimal = dc

    return dc_optimal


def dc_prediction_1(dct_coefs, dc_preset):
    """
    DC coefficients prediction from up-left to right-down.
    :param dct_coefs: ndarray, DCT coefficients of blocks
    :param dc_preset: float, preset dc value of reference block
    :return: dct_preds: ndarray, estimated DCT coefficients of blocks
    """

    dct_preds = np.zeros([w_n, h_n, patch_size, patch_size])
    dct_preds[0, 0, 0, 0] = dc_preset
    for i in range(w_n):
        for j in range(h_n):
            dct_target = dct_coefs[i, j]
            if i == 0 and j == 0:
                dct_target[0, 0] = dct_preds[i, j, 0, 0]
                dct_preds[i, j] = dct_target
                continue
            if j == 0:
                dct_up = dct_coefs[i - 1, j]
                dct_up[0, 0] = dct_preds[i - 1, j, 0, 0]
                dct_target[0, 0] = estimate(dct_target=dct_target, dct_up=dct_up)
            else:
                dct_left = dct_coefs[i, j - 1]
                dct_left[0, 0] = dct_preds[i, j - 1, 0, 0]
                if i == 0:
                    dct_target[0, 0] = estimate(dct_target=dct_target, dct_left=dct_left)
                else:
                    dct_up = dct_coefs[i - 1, j]
                    dct_up[0, 0] = dct_preds[i - 1, j, 0, 0]
                    dct_target[0, 0] = estimate(dct_target=dct_target, dct_up=dct_up, dct_left=dct_left)
            dct_preds[i, j] = dct_target

    return dct_preds


def dc_prediction_2(dct_coefs, dc_preset):
    """
    DC coefficients prediction from up-right to left-down.
    :param dct_coefs: ndarray, DCT coefficients of blocks
    :param dc_preset: float, preset dc value of reference block
    :return: dct_preds: ndarray, estimated DCT coefficients of blocks
    """

    dct_preds = np.zeros([w_n, h_n, patch_size, patch_size])
    dct_preds[0, h_n - 1, 0, 0] = dc_preset
    for i in range(w_n):
        for j in np.arange(h_n - 1, -1, -1):
            dct_target = dct_coefs[i, j]
            if i == 0 and j == h_n - 1:
                dct_target[0, 0] = dct_preds[i, j, 0, 0]
                dct_preds[i, j] = dct_target
                continue
            if j == h_n - 1:
                dct_up = dct_coefs[i - 1, j]
                dct_up[0, 0] = dct_preds[i - 1, j, 0, 0]
                dct_target[0, 0] = estimate(dct_target=dct_target, dct_up=dct_up)
            else:
                dct_right = dct_coefs[i, j + 1]
                dct_right[0, 0] = dct_preds[i, j + 1, 0, 0]
                if i == 0:
                    dct_target[0, 0] = estimate(dct_target=dct_target, dct_right=dct_right)
                else:
                    dct_up = dct_coefs[i - 1, j]
                    dct_up[0, 0] = dct_preds[i - 1, j, 0, 0]
                    dct_target[0, 0] = estimate(dct_target=dct_target, dct_up=dct_up, dct_right=dct_right)
            dct_preds[i, j] = dct_target

    return dct_preds


def dc_prediction_3(dct_coefs, dc_preset):
    """
    DC coefficients prediction from left-down to up-right.
    :param dct_coefs: ndarray, DCT coefficients of blocks
    :param dc_preset: float, preset dc value of reference block
    :return: dct_preds: ndarray, estimated DCT coefficients of blocks
    """

    dct_preds = np.zeros([w_n, h_n, patch_size, patch_size])
    dct_preds[w_n - 1, 0, 0, 0] = dc_preset
    for i in np.arange(w_n - 1, -1, -1):
        for j in range(h_n):
            dct_target = dct_coefs[i, j]
            if i == w_n - 1 and j == 0:
                dct_target[0, 0] = dct_preds[i, j, 0, 0]
                dct_preds[i, j] = dct_target
                continue
            if j == 0:
                dct_down = dct_coefs[i + 1, j]
                dct_down[0, 0] = dct_preds[i + 1, j, 0, 0]
                dct_target[0, 0] = estimate(dct_target=dct_target, dct_down=dct_down)
            else:
                dct_left = dct_coefs[i, j - 1]
                dct_left[0, 0] = dct_preds[i, j - 1, 0, 0]
                if i == w_n - 1:
                    dct_target[0, 0] = estimate(dct_target=dct_target, dct_left=dct_left)
                else:
                    dct_down = dct_coefs[i + 1, j]
                    dct_down[0, 0] = dct_preds[i + 1, j, 0, 0]
                    dct_target[0, 0] = estimate(dct_target=dct_target, dct_down=dct_down, dct_left=dct_left)
            dct_preds[i, j] = dct_target

    return dct_preds


def dc_prediction_4(dct_coefs, dc_preset):
    """
    DC coefficients prediction from right-down to up-left.
    :param dct_coefs: ndarray, DCT coefficients of blocks
    :param dc_preset: float, preset dc value of reference block
    :return: dct_preds: ndarray, estimated DCT coefficients of blocks
    """

    dct_preds = np.zeros([w_n, h_n, patch_size, patch_size])
    dct_preds[w_n - 1, h_n - 1, 0, 0] = dc_preset
    for i in np.arange(w_n - 1, -1, -1):
        for j in np.arange(h_n - 1, -1, -1):
            dct_target = dct_coefs[i, j]
            if i == w_n - 1 and j == h_n - 1:
                dct_target[0, 0] = dct_preds[i, j, 0, 0]
                dct_preds[i, j] = dct_target
                continue
            if j == h_n - 1:
                dct_down = dct_coefs[i + 1, j]
                dct_down[0, 0] = dct_preds[i + 1, j, 0, 0]
                dct_target[0, 0] = estimate(dct_target=dct_target, dct_down=dct_down)
            else:
                dct_right = dct_coefs[i, j + 1]
                dct_right[0, 0] = dct_preds[i, j + 1, 0, 0]
                if i == w_n - 1:
                    dct_target[0, 0] = estimate(dct_target=dct_target, dct_right=dct_right)
                else:
                    dct_down = dct_coefs[i + 1, j]
                    dct_down[0, 0] = dct_preds[i + 1, j, 0, 0]
                    dct_target[0, 0] = estimate(dct_target=dct_target, dct_down=dct_down, dct_right=dct_right)
            dct_preds[i, j] = dct_target

    return dct_preds


def recover_from_dct(dct_coefs, mode='compress'):
    """
    Recover image from DCT coefficients.
    :param dct_coefs: ndarray, DCT coefficients of blocks
    :param mode: string, compression or normal mode
    :return: image_rec: ndarray, recovered image
    """
    image_rec = np.zeros(image_shape)
    if mode == 'normal':
        for i in range(w_n):
            for j in range(h_n):
                idct = cv2.idct(dct_coefs[i, j])
                image_rec[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = idct + 128
    elif mode == 'compress':
        for i in range(w_n):
            for j in range(h_n):
                idct = cv2.idct(dct_coefs[i, j] * Q)
                image_rec[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = idct + 128

    return image_rec


def dc_recovery(image_path):
    """
    Recover image from four corners.
    :param image_path: string, path of image file
    :return: dct_preds: ndarray, predicted DCT coefficients of blocks
             image_rec: ndarray, recovered image
    """
    image = read_image(image_path)
    dct_coefs = dct_transform(image, mode='compress', dc_free=True)
    dct_coefs_corner = dct_transform(image, mode='compress', dc_free=False)
    dc_preset = [dct_coefs_corner[0, 0, 0, 0], dct_coefs_corner[0, h_n - 1, 0, 0],
                 dct_coefs_corner[w_n - 1, 0, 0, 0], dct_coefs_corner[w_n - 1, h_n - 1, 0, 0]]
    dct_preds = dc_prediction_1(dct_coefs, dc_preset[0])
    dct_preds += dc_prediction_2(dct_coefs, dc_preset[1])
    dct_preds += dc_prediction_3(dct_coefs, dc_preset[2])
    dct_preds += dc_prediction_4(dct_coefs, dc_preset[3])
    dct_preds /= 4.
    image_rec = recover_from_dct(dct_preds, mode='compress')

    return dct_preds, image_rec


def main():
    image_path = "../dataset/girlface.jpg"
    _, image_rec = dc_recovery(image_path)
    cv2.imwrite("../results/girlface_rec.jpg", image_rec)


if __name__ == '__main__':
    main()
