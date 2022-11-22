#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# version (0 for VGG and 1 for ResNet)
version = 0
# batch size
video_b_s = 2
# image_b_s = 20
image_b_s = 20
# number of frames
num_frames = 5

# number of cols of input images
shape_c = 320

if version == 1:
    # number of rows of input images
    shape_r = 256
    # number of rows of model outputs
    shape_r_out = 64
    # number of cols of model outputs
    shape_c_out = 80
else:
    # number of rows of input images
    shape_r = 256#240
    # number of cols of model outputs
    shape_c_out = 160
    # number of rows of model outputs
    shape_r_out = 128#120


# number of rows of attention
shape_r_attention = 64
# number of cols of attention
shape_c_attention = 80

# number of rows of downsampled maps
shape_r_gt = 32
# number of cols of downsampled maps
shape_c_gt = 40

# final upsampling factor
upsampling_factor = 16
# number of epochs
nb_epoch = 10
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16

# path of continuous saliency map
maps_path = 'groundtruth/'
# path of fixation maps
# fixs_path = '/fixation/maps/'
# path of images
frames_path = 'input/'

# number of training videos
# nb_train = 10800
# nb_train = 5000
# nb_train = 1000
nb_train = 1000
# number of validation videos
# nb_videos_val = 150
# nb_videos_val = 100
nb_videos_val = 100
realtime_frames = 3
#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training videos
videos_train_paths = ['/home/natnael/Documents/datasets/cdnet2014/training_ds/']
# path of validation videos
videos_val_paths = ['/home/natnael/Documents/datasets/cdnet2014/validation/']

videos_test_path = '../DHF1K/Videos/test_video/'
#
# # path of training maps
# maps_path = '/maps/'
# # path of training fixation maps
# fixs_path = '/fixation/maps/'
#
# frames_path = '/images/'

# path of training images
imgs_path = '/home/natnael/Documents/datasets/cdnet2014/training_ds/'
# imgs_path = 'D:/code/attention/staticimages/training/'