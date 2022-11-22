from __future__ import division

from collections import deque

from keras.layers import Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import os
import numpy as np
from config import *

import visualkeras
# from utilities import preprocess_images, preprocess_bin_images, preprocess_maps, preprocess_fixmaps, \
#     postprocess_predictions
# from models import acl_vgg, schedule_vgg, kl_divergence, correlation_coefficient, nss
# from scipy.misc import imread, imsave
import random
from math import ceil

import cv2 as cv
from utilities import *
from model import *


def generator(video_b_s, image_b_s, phase_gen='train'):
    if phase_gen == 'train':
        # videos = [videos_train_path + f for videos_train_path in videos_train_paths for f in
        #           os.listdir(videqos_train_path) if os.path.isfile(videos_train_path + f)]
        videos_train_paths = ['/home/natnael/Documents/datasets/cdnet2014/training_ds/']
        videos = get_video_shuffled(videos_train_paths)

        # print(videos[0])

        for item_video in videos:
            images = [item_video + frames_path + f for f in
                      os.listdir(item_video + frames_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            gt_images = [item_video + maps_path + f for f in
                         os.listdir(item_video + maps_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        #     # images = [video +'/' + f for video in videos for f in os.listdir(video+'/') if f.endswith(('.jpg', '.jpeg', '.png'))]
        #
        # # spatio_temporal_images = [imgs_path + 'residual_training/'+ video[-8:-4] +'/' + f for video in videos for f in os.listdir(imgs_path + 'residual_training/'+video[-8:-4] +'/') if f.endswith(('.jpg', '.jpeg', '.png'))]
        #
        # # nb_train = len(images)/image_b_s
        # # print(nb_train, len(images))
        # fixationmaps = [imgs_path + item_video[-4:] + '/maps/' + f for f in
        #                 os.listdir(imgs_path + item_video[-4:] + '/maps/') if f.endswith(('.jpg', '.jpeg', '.png'))]
        # # fixationmaps = [imgs_path +'annotation/0'+video[-3:] + '/maps/' + f for video in videos for f in os.listdir(imgs_path +'annotation/0'+video[-3:] +'/maps/') if f.endswith(('.jpg', '.jpeg', '.png'))]
        #
        # fixs = [imgs_path + item_video[-4:] + '/fixation/maps/' + f for video in videos for f in
        #         os.listdir(imgs_path + item_video[-4:] + '/fixation/maps/') if f.endswith(('.mat'))]

        images.sort()
        gt_images.sort()

        image_train_data = []
        for image, gt_image in zip(images, gt_images):
            annotation_data = {'image': image, 'gt_map': gt_image}  # changed
            image_train_data.append(annotation_data)

        random.shuffle(image_train_data)

        # videos.sort()
        random.shuffle(videos)

        loop = 1
        video_counter = 0
        image_counter = 0
        while True:
            # if loop % 2:
            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xims2 = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xims3 = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))

            Ymaps = np.zeros((video_b_s, num_frames, shape_r_out, shape_c_out, 1)) + 0.01

            Img_Ymaps = np.zeros((video_b_s, num_frames, shape_r_attention, shape_c_attention, 1)) + 0.01

            for i in range(0, video_b_s):
                video_path = videos[(video_counter + i) % len(videos)]
                # images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                #           f.endswith(('.jpg', '.jpeg', '.png'))]

                images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                          f.endswith(('.jpg', '.jpeg', '.png'))]

                # spatio_temporal = [imgs_path + 'residual_training/' + video_path[-7:-4] + '/' + f for f in
                #           os.listdir(imgs_path + 'residual_training/' + video_path[-7:-4] + '/') if
                #           f.endswith(('.jpg', '.jpeg', '.png'))]

                # print(len(images), len(spatio_temporal))

                # merger function like images = images,spatio_temporal -- split 3 and merge 4

                gt_images = [video_path + maps_path + f for f in os.listdir(video_path + maps_path) if
                             f.endswith(('.jpg', '.jpeg', '.png'))]

                images.sort()
                gt_images.sort()

                start = random.randint(0, max(len(images) - num_frames, 0))

                # [X] = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
                [X, X2, X3] = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
                # X3= preprocess_three_frame(images[start:min(start + num_frames, len(images))], shape_r, shape_c)

                Y = preprocess_maps(gt_images[start:min(start + num_frames, len(images))], shape_r_out, shape_c_out)

                Xims[i, 0:X.shape[0], :] = np.copy(X)
                Xims2[i, 0:X.shape[0], :] = np.copy(X2)
                Xims3[i, 0:X.shape[0], :] = np.copy(X3)

                Ymaps[i, 0:Y.shape[0], :] = np.copy(Y)

                Xims[i, X.shape[0]:num_frames, :] = np.copy(X[-1, :, :])
                Xims2[i, X.shape[0]:num_frames, :] = np.copy(X2[-1, :, :])
                Xims3[i, X.shape[0]:num_frames, :] = np.copy(X3[-1, :, :])

                Ymaps[i, Y.shape[0]:num_frames, :] = np.copy(Y[-1, :, :])

            # yield Xims, Ymaps  # add second input here
            yield [Xims, Xims2, Xims3], Ymaps  # add second input here
            video_counter = (video_counter + video_b_s) % len(videos)
            loop = loop + 1
            # else:
            #     Xims = np.zeros((image_b_s, 1, shape_r, shape_c, 3))
            #     Xims2 = np.zeros((image_b_s, 1, shape_r, shape_c, 3))
            #     # Xims3 = np.zeros((image_b_s, 1, shape_r, shape_c, 3))
            #
            #     Ymaps = np.zeros((image_b_s, 1, shape_r_out, shape_c_out, 1)) + 0.01
            #
            #     Img_Ymaps = np.zeros((image_b_s, 1, shape_r_attention, shape_c_attention, 1)) + 0.01
            #     Img_Yfixs = np.zeros((image_b_s, 1, shape_r_attention, shape_c_attention, 1)) + 0.01
            #
            #     for i in range(0, image_b_s):
            #         img_data = image_train_data[(image_counter + i) % len(image_train_data)]
            #         print(len(img_data), "from odd section")
            #         # spatio_temporal_data = image_train_data[(image_counter + i) % len(image_train_data)]
            #
            #         # X = preprocess_images([img_data['image']], shape_r, shape_c)
            #         [X, X2] = preprocess_images([img_data['image']], shape_r, shape_c)
            #         # X3 = preprocess_three_frame([img_data['image']], shape_r, shape_c)
            #
            #         Y = preprocess_maps([img_data['gt_map']], shape_r_attention, shape_c_attention)
            #
            #         Xims[i, 0, :] = np.copy(X)
            #         Xims2[i, 0, :] = np.copy(X2)
            #         # Xims3[i, 0, :] = np.copy(X3)
            #
            #         Img_Ymaps[i, 0, :] = np.copy(Y)
            #
            #     yield [Xims, Xims2], Ymaps  # add second input here
            #     image_counter = (image_counter + image_b_s) % len(image_train_data)
            #     loop = loop + 1

    elif phase_gen == 'val':
        videos_train_paths = ['/home/natnael/Documents/datasets/cdnet2014/validation/']
        videos = get_video_shuffled(videos_train_paths)

        print("validation size", len(videos))

        random.shuffle(videos)

        video_counter = 0
        while True:

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xims2 = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xims3 = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))

            Ymaps = np.zeros((video_b_s, num_frames, shape_r_out, shape_c_out, 1)) + 0.01

            Img_Ymaps = np.zeros((video_b_s, num_frames, shape_r_attention, shape_c_attention, 1)) + 0.01
            Img_Yfixs = np.zeros((video_b_s, num_frames, shape_r_attention, shape_c_attention, 1)) + 0.01

            for i in range(0, video_b_s):
                video_path = videos[(video_counter + i) % len(videos)]
                images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                          f.endswith(('.jpg', '.jpeg', '.png'))]

                gt_images = [video_path + maps_path + f for f in os.listdir(video_path + maps_path) if
                             f.endswith(('.jpg', '.jpeg', '.png'))]

                start = random.randint(0, max(len(images) - num_frames, 0))
                # X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
                [X, X2, X3] = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
                # X3= preprocess_three_frame(images[start:min(start + num_frames, len(images))], shape_r, shape_c)

                Y = preprocess_maps(gt_images[start:min(start + num_frames, len(images))], shape_r_out, shape_c_out)

                Xims[i, 0:X.shape[0], :] = np.copy(X)
                Xims2[i, 0:X2.shape[0], :] = np.copy(X2)
                Xims3[i, 0:X3.shape[0], :] = np.copy(X3)

                Ymaps[i, 0:Y.shape[0], :] = np.copy(Y)

                Xims[i, X.shape[0]:num_frames, :] = np.copy(X[-1, :, :])
                Xims2[i, X2.shape[0]:num_frames, :] = np.copy(X2[-1, :, :])
                Xims3[i, X3.shape[0]:num_frames, :] = np.copy(X3[-1, :, :])

                Ymaps[i, Y.shape[0]:num_frames, :] = np.copy(Y[-1, :, :])

            # yield Xims, Ymaps
            yield [Xims, Xims2, Xims3], Ymaps
            video_counter = (video_counter + video_b_s) % len(videos)
    else:
        raise NotImplementedError

def get_test(video_test_path, iterations):
    print("Video Test Path - ", video_test_path)
    images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
              f.endswith(('.jpg', '.jpeg', '.png'))]

    # print("from get test", len(images))
    images.sort()
    start = 0
    while True:
        Xims = np.zeros((1, num_frames, shape_r, shape_c, 3))  # change dimensionality
        Xims2 = np.zeros((1, num_frames, shape_r, shape_c, 3))  # change dimensionality
        Xims3 = np.zeros((1, num_frames, shape_r, shape_c, 3))  # change dimensionality
        # Xims2 = np.zeros((1, num_frames, shape_r, shape_c, 3))  # change dimensionality
        # print("Shape Xims", Xims.shape)

        [X, X2, X3] = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        # X2 = preprocess_bin_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)

        Xims[0, 0:min(len(images) - start, num_frames), :] = np.copy(X)
        Xims2[0, 0:min(len(images) - start, num_frames), :] = np.copy(X2)
        Xims3[0, 0:min(len(images) - start, num_frames), :] = np.copy(X3)
        # Xims2[0, 0:min(len(images) - start, num_frames), :] = np.copy(X2)

        yield [Xims, Xims2, Xims3]
        # print(Xims.shape)
        start = min(start + num_frames, len(images))


def get_test_realtime(images):
    # print("Video Test Path - ", video_test_path)
    # images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
    #           f.endswith(('.jpg', '.jpeg', '.png'))]

    # print("from get test", len(images))
    # images.sort()
    start = 0
    while True:
        Xims = np.zeros((1, len(images), shape_r, shape_c, 3))  # change dimensionality
        Xims2 = np.zeros((1, len(images), shape_r, shape_c, 3))  # change dimensionality
        Xims3 = np.zeros((1, len(images), shape_r, shape_c, 3))  # change dimensionality
        # Xims2 = np.zeros((1, num_frames, shape_r, shape_c, 3))  # change dimensionality
        # print("Shape Xims", Xims.shape)

        [X, X2, X3] = preprocess_images_realtime(images, shape_r, shape_c)
        # X2 = preprocess_bin_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        print("Inside Test Generator ", X.shape)
        Xims[0, 0:max(len(images) - start, len(images)), :] = np.copy(X)
        Xims2[0, 0:max(len(images) - start, len(images)), :] = np.copy(X2)
        Xims3[0, 0:max(len(images) - start, len(images)), :] = np.copy(X3)
        # Xims2[0, 0:min(len(images) - start, num_frames), :] = np.copy(X2)

        yield [Xims, Xims2, Xims3]
        # print(Xims.shape)
        start = min(start + num_frames, len(images))


if __name__ == '__main__':
    phase = 'test'
    if phase == 'train':

        x = Input(shape=(None, shape_r, shape_c, 3))
        x2 = Input(shape=(None, shape_r, shape_c, 3))
        x3 = Input(shape=(None, shape_r, shape_c, 3))
        # x2 = Input(batch_shape=(None, None, shape_r, shape_c, 3))
        stateful = False
    else:
        # x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
        x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
        x2 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
        x3 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
        stateful = True

    if phase == 'train':
        if nb_train % video_b_s != 0 or nb_videos_val % video_b_s != 0:
            print("The number of training and validation images should be a multiple of the batch size. "
                  "Please change your batch size in config.py accordingly.")
            exit()

        # m = Model(inputs=[x, x2, x3], outputs=transform_saliency([x, x2, x3], stateful))
        m = Model(inputs=[x, x2, x3], outputs=transform_saliency([x, x2, x3], stateful))

        # visualkeras.layered_view(m).show()  # display using your system viewer
        # visualkeras.layered_view(m, to_file='output.png')  # write to disk
        # visualkeras.layered_view(m, to_file='output.png').show()  # write and show

        print("Compiling XYshift ...")
        m.compile(Adam(lr=1e-4),
                  loss=['binary_crossentropy'])
        print("Training ACL-VGG")

        m.fit_generator(generator(video_b_s=video_b_s, image_b_s=image_b_s), nb_train, epochs=nb_epoch,
                        validation_data=generator(video_b_s=video_b_s, image_b_s=0, phase_gen='val'),
                        validation_steps=nb_videos_val,
                        callbacks=[EarlyStopping(patience=15),
                                   ModelCheckpoint('xy-shift.{epoch:02d}-{val_loss:.4f}.h5', save_best_only=True),
                                   LearningRateScheduler(schedule=schedule_vgg)])

        m.save('XYshift.h5')

    # elif phase == "test":
    #     # Crosscheck and adjust
    #     # videos_test_path = '../DHF1K/test_imgs/'
    #     result_path = '/home/natnael/Documents/datasets/cdnet2014/test/'
    #     # videos_test_path = '../DHF1K/val_images/'
    #     videos_test_path = '/home/natnael/Documents/datasets/cdnet2014/test/'
    #     # videos = [videos_test_path + f for f in os.listdir(videos_test_path) if os.path.isdir(videos_test_path + f)]
    #
    #     videos = [videos_test_path + sub_path1 + '/' + sub_path2 + '/' for sub_path1 in os.listdir(videos_test_path) for sub_path2 in
    #                     os.listdir(videos_test_path + sub_path1)]
    #     # print(videos[0], "first video")
    #
    #     print(len(videos))
    #     # for i in videos:
    #     #     print(i)
    #     videos.sort()
    #
    #     nb_videos_test = len(videos)
    #     print(len(videos), "length of videos")
    #
    #     # m = Model(inputs=[x, x2, x3], outputs=xy_shift([x, x2, x3], stateful))  # change this later
    #     # m = Model(inputs=x, outputs=xy_shift(x, stateful))  # change this later
    #     m = Model(inputs=[x, x2, x3], outputs=transform_saliency([x, x2, x3], stateful))
    #     print("Loading XYshift weights")
    #     # m = Model(inputs=x, outputs=acl_vgg(x, stateful))
    #     # print("Loading ACL weights")
    #
    #     m.load_weights('XYshift.h5')
    #
    #     print("loaded model weight")
    #     # print(m.summary())
    #     # for i in range(25, nb_videos_test):
    #     for i in range(nb_videos_test):
    #
    #         # print(videos[i])
    #         # print(videos[i])
    #
    #         output_folder = videos[i] + '/detection_folder/'
    #         # print(output_folder)
    #         if not os.path.exists(output_folder):
    #             os.makedirs(output_folder)
    #         images_names = [f for f in os.listdir(videos[i] + frames_path) if
    #                         f.endswith(('.jpg', '.jpeg', '.png'))]
    #         print(len(images_names), "total images per video inside test")
    #
    #         images_names.sort()
    #         print(len(images_names), "Image count")
    #         # print(images_names[0])
    #         # print(len(images_names)
    #
    #         print("Classifying moving pixels for " + videos[i])
    #         # prediction = m.predict_generator(get_test(video_test_path=videos[i]),max(ceil(len(images_names) / num_frames), 2))
    #         prediction = m.predict_generator(get_test(video_test_path=videos[i]),1)
    #         predictions = prediction[0]
    #         print("Reached here, iterating images")
    #         print(len(images_names))
    #         for j in range((len(images_names)-7989)):
    #             original_image = cv.imread(videos[i] + frames_path + images_names[j])
    #             x, y = divmod(j, num_frames)
    #
    #             print(predictions.shape)
    #             print(prediction.shape)
    #
    # #            res = postprocess_predictions(predictions[x, y, :, :, 0], original_image.shape[0],
    #                                           original_image.shape[1])
    #
    #             # cv.imshow("Output", res)
    #             cv.imwrite(output_folder + '%s' % images_names[j], res.astype(int))
    #             # imsave(output_folder + '%s' % images_names[j], res.astype(int))
    #         m.reset_states()
    # else:
    #     raise NotImplementedError

    elif phase.__contains__("test"):
        #realtime test
        queue = deque()
        m = Model(inputs=[x, x2, x3], outputs=transform_saliency([x, x2, x3], stateful))

        print("Loading ACL weights")
        m.load_weights('XYshift.h5')

        # Get access to local camera
        counter = 1 # just for debug purose

        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            if len(queue) != realtime_frames:
                queue.append(frame)
                continue
            else:
                queue.popleft()
                queue.append(frame)

            Xims = np.zeros((1, len(queue), shape_r, shape_c, 3))  # change dimensionality
            Xims2 = np.zeros((1, len(queue), shape_r, shape_c, 3))  # change dimensionality
            Xims3 = np.zeros((1, len(queue), shape_r, shape_c, 3))  # change dimensionality

            [X, X2, X3] = preprocess_images_realtime(queue, shape_r, shape_c)
            print(X.shape, "X shape new")
            # print("Inside Test Generator ", X.shape)
            Xims[0] = np.copy(X)
            Xims2[0] = np.copy(X2)
            Xims3[0] = np.copy(X3)

            prediction = m.predict([Xims, Xims2, Xims3])
            print(prediction.shape, "Predicted mode")
            print(counter)
            counter = counter + 1
            for j in range(len(queue)):
                original_image = queue[0]
                print(original_image.shape, "- Queue shape")
                x, y = divmod(j, len(queue))
                print(x, y)

               #pos process to scale up, if required
                print(prediction[0, 0, :, :, 0].shape)

                cv.imshow("Frame", prediction[0, 0, :, :, 0])
                cv.waitKey(3)
            # reset states for the next round
            m.reset_states()



        # Generate for evaluation - Recall, Precsion, F1-Score
        # videos_test_path = '../DHF1K/test_imgs/'
        # result_path = '/home/natnael/Documents/datasets/cdnet2014/test/'
        # videos_test_path = '../DHF1K/val_images/'
        # videos_test_path = '/home/natnael/Documents/datasets/cdnet2014/test/'
        # videos = [videos_test_path + sub_path1 + '/' + sub_path2 + '/' for sub_path1 in os.listdir(videos_test_path) for
        #           sub_path2 in os.listdir(videos_test_path + sub_path1)]
        # videos = [videos_test_path + f for f in os.listdir(videos_test_path) if os.path.isdir(videos_test_path + f)]

        # for i in videos:
        #     print(i)
        # videos.sort()

        # nb_videos_test = len(videos)
        # print(videos[99])

        # for i in range(0, nb_videos_test): # was 25 to nb  _vieos_test
        #     # print(videos[i])
        #     # print(videos[i])
        #
        #     output_folder = videos[i] + '/detection_folder/'
        #     if not os.path.exists(output_folder):
        #         os.makedirs(output_folder)
        #
        #     images_names = [f for f in os.listdir(videos[i] +frames_path) if
        #                     f.endswith(('.jpg', '.jpeg', '.png'))]
        #
        #
        #     images_names.sort()
        #
        #     print(len(images_names), "Image count")
        #     # print(images_names[0])
        #     # print(len(images_names))
        #
        #
        #     print("Predicting saliency maps for " + videos[i])
        #     print(max(ceil(len(images_names) / num_frames), 2))
        #
        #     iteration = max(ceil(len(images_names) / num_frames), 2)        #

        #     prediction = m.predict_generator(get_test(video_test_path=videos[i], iterations=iteration),
        #     max(ceil(len(images_names) / num_frames), 2))
        #     predictions = prediction[0]
        #     print(prediction.shape)
        #     print(predictions.shape)
        #
        #
        #     for j in range(len(images_names)):
        #         original_image = cv.imread(videos[i] + frames_path + images_names[j])
        #         x, y = divmod(j, num_frames)
        #         res = postprocess_predictions(prediction[x, y, :, :, 0], original_image.shape[0],
        #                                       original_image.shape[1])

        #         cv.imwrite(output_folder + '%s' % images_names[j], res.astype(int))
        #         # imsave(output_folder + '%s' % images_names[j], res.astype(int))
        #     m.reset_states()
    else:
        raise NotImplementedError
