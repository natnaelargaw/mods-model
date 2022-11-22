from __future__ import division

import os

import numpy as np
import scipy.io
import scipy.ndimage
# from scipy.misc import imresize
import cv2 as cv
import numpy as np
import random

from config import shape_c, shape_r, frames_path, num_frames


def preprocess_images(paths, shape_r, shape_c):

    # print(len(paths))
    # for i in paths:
    #     print(i)

    ims_numpy = np.zeros((len(paths), shape_r, shape_c, 3))
    ims2_numpy = np.zeros((len(paths), shape_r, shape_c, 3))
    ims3_numpy = np.zeros((len(paths), shape_r, shape_c, 3))
    ims =[]
    ims2 = []
    # ims3 = []
    # ims4 = []

    for i, ori_path in enumerate(paths):
        original_image = cv.imread(ori_path)
        copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
        if original_image.shape == 2:
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image

            original_image = copy

        padded_image = padding(original_image, shape_r, shape_c, 3)



        ims_numpy[i] = padded_image
        ims.append(padded_image)

        # cv.imshow("Frame", ims[i]/255)
        # cv.waitKey(0)


        ims2_numpy[i] = cv.addWeighted(padded_image,0.1 ,xy_shift(padded_image),0.8,0) # weighted xyshift

        # cv.imshow("Frame", ims2_numpy[i]/255)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        ims2.append(xy_shift(padded_image))

    ims3 =improved_inter_frame_differencing(ims2).copy()
    for i, img in enumerate(ims3):
        ims3_numpy[i] = img

        intermediate_image = cv.addWeighted(ims_numpy[i],0.3, ims2_numpy[i], 0.7,0)
        ims3_numpy[i] = cv.addWeighted(intermediate_image,0.7, ims3_numpy[i], 0.7,0)
        # last_image = cv.addWeighted(intermediate_image,0.7, ims3_numpy[i], 0.7,0)
        # cv.imshow("Frame", last_image/ 255)
        # cv.imshow("Frame_3", ims3_numpy[i]/ 255)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    return [ims_numpy,ims2_numpy, ims3_numpy]
    # return [ims]




def preprocess_images_realtime(paths, shape_r, shape_c): #paths are images here

    # print(len(paths))
    # for i in paths:
    #     print(i)
    ims_numpy = np.zeros((len(paths), shape_r, shape_c, 3))
    ims2_numpy = np.zeros((len(paths), shape_r, shape_c, 3))
    ims3_numpy = np.zeros((len(paths), shape_r, shape_c, 3))
    ims =[]
    ims2 = []


    for i, original_image in enumerate(paths):
        copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
        if original_image.shape == 2:
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image

            original_image = copy

        padded_image = padding(original_image, shape_r, shape_c, 3)


        ims_numpy[i] = padded_image
        ims.append(padded_image)

        # cv.imshow("Frame", ims[i]/255)
        # cv.waitKey(0)


        ims2_numpy[i] = cv.addWeighted(padded_image,0.1 ,xy_shift(padded_image),0.8,0) # weighted xyshift

        # cv.imshow("Frame", ims2_numpy[i]/255)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        ims2.append(xy_shift(padded_image))

    ims3 =improved_inter_frame_differencing(ims2).copy()
    for i, img in enumerate(ims3):
        ims3_numpy[i] = img

        intermediate_image = cv.addWeighted(ims_numpy[i],0.3, ims2_numpy[i], 0.7,0)
        ims3_numpy[i] = cv.addWeighted(intermediate_image,0.7, ims3_numpy[i], 0.7,0)
        # last_image = cv.addWeighted(intermediate_image,0.7, ims3_numpy[i], 0.7,0)
        # cv.imshow("Frame", last_image/ 255)
        # cv.imshow("Frame_3", ims3_numpy[i]/ 255)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    return [ims_numpy,ims2_numpy, ims3_numpy]

























def merge_image(paths, a,b,c):
    ims = np.zeros((len(paths), shape_r, shape_c * 3, 3))
    # print(ims.shape)

    for i_0,(i,j,k) in enumerate(zip(a,b,c)):
        inter_image = np.concatenate((i, j), axis=1)
        outs = np.concatenate((inter_image, k), axis=1)
        ims[i_0] = outs
        # print("New image size", outs.shape)
        # cv.imshow("Frame", outs)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    return ims
def preprocess_three_frame(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, ori_path in enumerate(paths):
        original_image = cv.imread(ori_path)
        copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
        if original_image.shape == 2:
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image

            original_image = copy

        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image
        # Potentially average of frames in each channel and every video
    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims[:, :, :, ::-1]

    ims = []
    # ims = three_frame_differencing(ims)


    return ims

def three_frame_differencing_modified(images):

    if len(images)== 2:
        images.append(images[1])
    frame_current = images[1]
    frame_ref_left = images[0]
    frame_ref_right = images[2]
    three_framed = transform_modified(frame_current, frame_ref_left, frame_ref_right)


    return three_framed


def three_frame_differencing(images):
    # print(len(small_batch), "Size in process batch")
    # frame_differenced_ims = np.zeros((len(images), shape_r, shape_c, 3))
    frame_differenced_ims = []

    for i in range(len(images)):
        if i == 0:
            frame_current = images[i]
            frame_ref_left = images[i + 1]
            frame_ref_right = images[i + 1]
            # cv.imshow("Frame 1", frame_current)
            # cv.waitKey(1)
        elif i == len(images)-1:#last element

            frame_current = images[i]
            frame_ref_left = images[i - 1]
            frame_ref_right = images[i - 2]
        else:
            frame_current = images[i]
            frame_ref_left = images[i - 1]
            frame_ref_right = images[i + 1]
        frame_differenced_ims.append(transform(frame_current, frame_ref_left, frame_ref_right))

    return frame_differenced_ims
def improved_inter_frame_differencing(images):
    # print(len(small_batch), "Size in process batch")
    # frame_differenced_ims = np.zeros((len(images), shape_r, shape_c, 3))
    frame_differenced_ims = []

    for i in range(len(images)):
        if i == 0:
            frame_current = images[i]
            frame_ref_left = images[i + 1]
            frame_ref_right = images[i + 1]
            # cv.imshow("Frame 1", frame_current)
            # cv.waitKey(1)
        else:
            frame_current = images[i]
            frame_ref_left = images[i - 1]
            frame_ref_right = images[i - 1]
        frame_differenced_ims.append(transform(frame_current, frame_ref_left, frame_ref_right))

    return frame_differenced_ims



def transform_modified(fc, fl,fr):
    delta_future = cv.absdiff(fr, fc)
    delta_past = cv.absdiff(fl, fc)

    r_future,g_future,b_future = cv.split(delta_future)
    r_past,g_past,b_past = cv.split(delta_past)

    r_current = np.maximum(r_future, r_past)
    g_current = np.maximum(g_future, g_past)
    b_current = np.maximum(b_future, b_past)

    out = cv.merge((r_current,g_current,b_current))
    # out = np.maximum(delta_future,delta_past)
    # out = cv.merge((out,out,out))
    return out




def transform(fc, fl,fr):
    # converting to grey - cv.cvtcolor doesn't work because of the numpy nature of image ?
    # rgb_weights = [0.2989, 0.5870, 0.01140]
    # frame_current = np.dot(fc[...,:3],rgb_weights)
    # frame_past = np.dot(fl[...,:3],rgb_weights)
    # frame_next = np.dot(fr[...,:3],rgb_weights)


    # delta_future = cv.absdiff(frame_next, frame_current)
    # delta_future = cv.absdiff(fc, fl)
    # delta_past = cv.absdiff(frame_past, frame_current)

    inter_frame =cv.absdiff(fc, fl)

    # r_future,g_future,b_future = cv.split(delta_future)
    # r_past,g_past,b_past = cv.split(delta_past)
    #
    # r_current = np.maximum(r_future, r_past)
    # g_current = np.maximum(g_future, g_past)
    # b_current = np.maximum(b_future, b_past)
    #
    # out = cv.merge((r_current,g_current,b_current))
    # out = np.maximum(delta_future,delta_past)
    # out = cv.merge((out,out,out))
    return inter_frame

def padding(img, shape_r=240, shape_c=320, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = imresize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols), ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = imresize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r

        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def imresize(image, shape=(224, 224)):
    img = cv.resize(image, shape)
    return img


def xy_shift(appearance):
    n = 2
    m = 4
    minus_m = 2

    subtrahend = appearance.copy()
    subtrahend[0:appearance.shape[0] - n, 0:appearance.shape[1] - n] = appearance[n:appearance.shape[0],
                                                                       n:appearance.shape[1]]

    subtrahend[appearance.shape[0] - n:, appearance.shape[1] - n:] \
        = \
        appearance[appearance.shape[0] - m:appearance.shape[0] - minus_m,
        appearance.shape[1] - m: appearance.shape[1] - minus_m]

    abs_diff = cv.absdiff(appearance, subtrahend)
    # cv.imshow("XY Frames", abs_diff)
    # cv.imshow("Raw Frames", raw_image)
    # cv.waitKey(1)

    return abs_diff


def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]
    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0] * factor_scale_r))
        c = int(np.round(coord[1] * factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out


def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols), ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def pass_view(ims):
    # frame = cv.imread("samples/450frames0.png")
    # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imwrite("samples/frames.png", grey)
    window_name = "Frame"
    for i in range(len(ims)):
        image = ims[i]
        cv.imshow(window_name, image)
        cv.waitKey(0)
        cv.destroyWindow(window_name)
        # change this image to grey
        # other methods do not work
        # rgb_weights = [0.2989, 0.5870, 0.01140]
        # grey = np.dot(image[...,:3],rgb_weights)
        # cv.imwrite("samples/"+str(len(ims))+"frames"+str(i)+".png", image)


# def preprocess_images(paths, shape_r, shape_c):
#     ims = np.zeros((len(paths), shape_r, shape_c, 3))
#
#     for i, path in enumerate(paths):
#         # print(path)
#         original_image = cv.imread(path)
#         # original_image = mpimg.imread(path)
#         # original_image = imread(path)
#         # if original_image.ndim == 2:
#         if original_image.shape == 2:
#             copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
#
#             copy[:, :, 0] = original_image
#             copy[:, :, 1] = original_image
#             copy[:, :, 2] = original_image
#
#             original_image = copy
#         padded_image = padding(original_image, shape_r, shape_c, 3)
#         ims[i] = padded_image
#
#     ims[:, :, :, 0] -= 103.939
#     ims[:, :, :, 1] -= 116.779
#     ims[:, :, :, 2] -= 123.68
#     ims = ims[:, :, :, ::-1]
#     # ims = ims.transpose((0, 3, 1, 2))
#
#     return ims


def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        original_map = cv.imread(path, 0)
        # original_map = mpimg.imread(path)
        # original_map = imread(path)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, :, :, 0] = padded_map.astype(np.float32)
        ims[i, :, :, 0] /= 255.0
    return ims


# def preprocess_fixmaps(paths, shape_r, shape_c):
#     ims = np.zeros((len(paths), shape_r, shape_c, 1))
#
#     for i, path in enumerate(paths):
#         fix_map = scipy.io.loadmat(path)["I"]
#         ims[i, :, :, 0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)
#
#     return ims


def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    pred = pred / np.max(pred) * 255

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        # pred = cv2.resize(pred, (new_cols, shape_r))
        pred = imresize(pred, (shape_r, new_cols))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        # pred = cv2.resize(pred, (shape_c, new_rows))
        pred = imresize(pred, (new_rows, shape_c))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    # img = scipy.ndimage.filters.gaussian_filter(img, sigma=7)
    img = img / np.max(img) * 255

    return img


# def merge_channels(rgb, binarized):
#     b, g, r = cv.split(rgb)
#     return cv.merge(b, g, r, binarized)






def get_video_shuffled(videos_train_paths):
    input_videos = [videos_train_path + sub_path1 + '/' + sub_path2 + '/' for videos_train_path in
                    videos_train_paths for sub_path1 in os.listdir(videos_train_path) for sub_path2 in
                    os.listdir(videos_train_path + sub_path1)]
    input_videos.sort()

    return input_videos


# def get_data_shuffled():
#     videos_train_paths = ['/home/natnael/Documents/datasets/cgnet2014/training_ds/']
#     input_frames = [videos_train_path + sub_path1 + '/' + sub_path2 + '/' + 'input/' + f for videos_train_path in
#                     videos_train_paths
#                     for sub_path1 in os.listdir(videos_train_path) for sub_path2 in
#                     os.listdir(videos_train_path + sub_path1)
#                     for f in os.listdir(videos_train_path + sub_path1 + '/' + sub_path2 + '/' + 'input/') if
#                     f.endswith(('.jpg', '.jpeg', '.png'))]
#
#     gt_frames = [videos_train_path + sub_path1 + '/' + sub_path2 + '/' + 'groundtruth/' + f for videos_train_path in
#                  videos_train_paths
#                  for sub_path1 in os.listdir(videos_train_path) for sub_path2 in
#                  os.listdir(videos_train_path + sub_path1)
#                  for f in os.listdir(videos_train_path + sub_path1 + '/' + sub_path2 + '/' + 'input/') if
#                  f.endswith(('.jpg', '.jpeg', '.png'))]
#     input_frames.sort()
#     gt_frames.sort()
#
#     image_train_data = []
#     for input_frame, gt_frame in zip(input_frames, gt_frames):
#         annotation_data = {'input': input_frame, 'gt': gt_frame}  # changed
#         image_train_data.append(annotation_data)
#
#     random.shuffle(image_train_data)
#     # print(image_train_data[10])
#     return image_train_data



def resizeandpad(img):
    copy = np.zeros((img.shape[0], img.shape[1], 3))
    original_image = img
    if original_image.shape == 2:
        # copy = np.zeros((original_image.shape[0], original_image.shape[1], 3)
        copy[:, :, 0] = original_image
        copy[:, :, 1] = original_image
        copy[:, :, 2] = original_image

        original_image = copy
    # if original_image.shape[2] ==3:
    #     original_image = merge_channels(original_image, bin_image)
    padded_image = padding(original_image, shape_r, shape_c, 3)
    return padded_image



def preprocess(path):
    img = cv.imread(path)
    out_image = resizeandpad(img)
    return out_image

if __name__ == '__main__':
    videos_train_paths = ['/home/natnael/Documents/datasets/cdnet2014/training_ds/']
    data = get_video_shuffled(videos_train_paths)
    for i in data:
        # print(i)
        images = [i + frames_path + j  for j in os.listdir(i + frames_path) if j.endswith(('.jpg', '.jpeg', '.png'))]
        for counter,j in enumerate(images):
            frame = cv.imread(j)


            xyshift = xy_shift(frame)

            last_image = cv.addWeighted(frame, 0.1,xyshift,1.0,0)
            # three_frame = three_frame_differencing_modified(images[max(0, counter - 1):min(counter + 2, len(images))])





            # cv.imshow("Frames3", three_frame)
            cv.imshow("Frames", last_image)
            cv.waitKey(1)
            print(j)



        # # print(len(images))
        # # images.sort()
        # # print(images[0])
        # # original_image = cv.imread(images[0])
        # # print(original_image.shape)
        #
        # start = random.randint(0, max(len(images) - num_frames, 0))
        #
        # # [X, X2, X3, X4] = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        # # X3 = improved_inter_frame_differencing(X2)
        # for x,y,z,a in zip(X,X2,X3, X4):
        #     # print(len(X))
        #     # print(len(X3))
        #     cv.imshow("Frames-raw iamges", x)
        #     cv.imshow("Frames2-xyshift", y)
        #     cv.imshow("Frame3 - interframe", z)
        #     cv.imshow("Frame4-threeframe", a)
        #     cv.waitKey()


