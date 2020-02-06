## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# This code has been modified to capture one still frame of information and save it to a file
#This will also briefly show the depth_image and

# First import the library, if you need this run "pip install pyrealsense2"
import pyrealsense2 as rs
# Import Numpy for easy array manipulation, if you need this run "pip install numpy"
import numpy as np
# Import OpenCV for easy image rendering, if you need this run "pip install opencv-python"
import cv2
# Import h5py to compress files into hdf5 format for dataset, if you need this run "pip install h5py"
import h5py
# import os to perform os.path calls for file i/o
import os
# import image lib to save images
from PIL import Image

TRAINX_SINGLE_DATA_SIZE = 307200;

def invertDepth(depth_image):
    largest = 0.0;
    depth_image_inverted = [];

    for x in range(len(depth_image)):
        for i in range(len(depth_image[x])):
            if (float(depth_image[x][i]) > largest):
                largest = float(depth_image[x][i]);

    for x in range(len(depth_image)):
        temp = [];
        for i in range(len(depth_image[x])):
            if (depth_image[x][i] != 0.0):
                temp.append(largest - depth_image[x][i]);
                depth_image[x][i] = temp[i];
            else:
                temp.append(depth_image[x][i]);
                depth_image[x][i] = temp[i];
        depth_image_inverted.append(temp);

    return np.asanyarray(depth_image_inverted);

def filterDepth(depth_image):

    smallest = 50000.0;
    depth_image_filtered = [];

    for x in range(len(depth_image)):
        for i in range(len(depth_image[x])):
            if (float(depth_image[x][i]) < smallest and float(depth_image[x][i]) != 0):
                smallest = float(depth_image[x][i]);

    if (smallest == 50000.0):
        smallest = 0.0;
    
    for x in range(len(depth_image)):
        temp = [];
        for i in range(len(depth_image[x])):
            if (depth_image[x][i] != 0.0):
                temp.append(depth_image[x][i] - smallest);
                depth_image[x][i] = temp[i];
            else:
                temp.append(depth_image[x][i]);
                depth_image[x][i] = temp[i];
        depth_image_filtered.append(temp);

    return np.asanyarray(depth_image_filtered);

# function used to normalize data from all real numbers to [0,1]
def normalizeDepthImage(depth_image):

    largest = 0.0;
    depth_image_normalized = [];

    for x in range(len(depth_image)):
        for i in range(len(depth_image[x])):
            if (float(depth_image[x][i]) > largest):
                largest = float(depth_image[x][i]);

    for x in range(len(depth_image)):
        temp = [];
        for i in range(len(depth_image[x])):
            if (largest != 0.0):
                temp.append(float(depth_image[x][i]) / float(largest));
                depth_image[x][i] = temp[i];
        depth_image_normalized.append(temp);

    return np.asanyarray(depth_image_normalized);

# function that outputs training data from camera to dataset.
def outputData(depth_image):
    print("Outputting Depth Image\n")
    # Writing single image file
    dset = {};

    depth_image_flat = [];
    for x in depth_image:
        for y in x:
            depth_image_flat.append(y);
    depth_image_flat = np.asanyarray(depth_image_flat);

    if ((os.path.isfile('datasets/train_x.hdf5'))):
        print("Appending to train_x dataset");
        with h5py.File('datasets/train_x.hdf5', 'a') as train_x:
            train_x["train_x"].resize((train_x["train_x"].shape[0] + depth_image_flat.shape[0]), axis=0);
            train_x["train_x"][-depth_image_flat.shape[0]:] = depth_image_flat;
    else:
        print("Creating new train_x dataset");
        with h5py.File('datasets/train_x.hdf5', 'w') as train_x:
            dset = train_x.create_dataset("train_x", data=depth_image_flat, compression="gzip", chunks=True, maxshape=(None,));

# function that outputs classification from GUI to dataset.
def outputClassification(text):
    #Convert text to dataset for append.
    textAsDataSet = np.asanyarray([text.lower().encode("ascii","ignore")]);

    if (os.path.isfile('datasets/train_y.hdf5')):
        print("Appending to train_y dataset");
        with h5py.File('datasets/train_y.hdf5', 'a') as train_y:
            train_y["train_y"].resize((train_y["train_y"].shape[0] + textAsDataSet.shape[0]), axis=0);
            train_y["train_y"][-textAsDataSet.shape[0]:] = textAsDataSet;
    else:
        print("Creating train_y dataset");
        with h5py.File('datasets/train_y.hdf5', 'w') as train_y:
            train_y.create_dataset("train_y", data=textAsDataSet, compression="gzip", dtype='S10', maxshape=(None,));

# function that gathers camera image data, displays it, and asks for classification
def gatherCameraImage():
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))
    cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Align Example', images)
    key = cv2.waitKey(1)

    #Invert depth image
    depth_image = invertDepth(depth_image);
    #Filter depth image
    depth_image = filterDepth(depth_image);
    #Normalize depth image
    depth_image = normalizeDepthImage(depth_image);

    #Writing depth image to file
    outputData(depth_image);

    pipeline.stop()

def collectTrainingX():
    if (os.path.exists('./datasets/train_x.hdf5')):
        with h5py.File('./datasets/train_x.hdf5', 'r') as train_x_file:
            train_x = train_x_file['train_x'];
            train_x_data = np.reshape(train_x, (int(len(train_x)/TRAINX_SINGLE_DATA_SIZE), TRAINX_SINGLE_DATA_SIZE));
            reshaped_x = [];
            for i in train_x_data:
                reshaped_x.append(np.reshape(i, (480, 640)));
            return reshaped_x;

def collectTrainingY():
    if (os.path.exists('./datasets/train_y.hdf5')):
        with h5py.File('./datasets/train_y.hdf5', 'r') as train_y_file:
            train_y = train_y_file['train_y'];
            train_y_data = [];
            for y in train_y:
                train_y_data.append(y.decode("utf-8"));
            return train_y_data;

def collectTestingX():
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))

    outputDataToFileStructure(depth_colormap, color_image);
    #Invert depth image
    depth_image = invertDepth(depth_image);
    #Filter depth image
    depth_image = filterDepth(depth_image);
    #Normalize depth image
    depth_image = normalizeDepthImage(depth_image);


    pipeline.stop()

    return depth_image;

def outputDataToFileStructure(depth_image, color_image):
    #now = datetime.now()  # current date and time
    #date_time = now.strftime("%m_%d_%Y__%H_%M_%S")
    # print("depth_image: ", depth_image)
    # print("text: ", label)
    # file_path = 'trainingSets/' # + label

    # if not os.path.exists(file_path):
    #     print("Creating filepath: ", file_path)
    #     os.makedirs(file_path)
    # else:
    #     print(file_path, " already exists")

    #Arbitrary name to be overwritten for testing purposes
    # file_path = file_path + '/' + 'color_test.png' 
    img = Image.fromarray(color_image, 'RGB')
    img.save("rgb_image.jpg")
    img = Image.fromarray(depth_image, 'RGB')
    img.save("depth_image.jpg")