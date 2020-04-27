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
# import os to perform os.path calls for file i/o
import os
# import image lib to save images
from PIL import Image

# function that outputs training data from camera to dataset.
def outputData(depth_colormap, gesture):
    strPath = '../datasets/' + gesture + '/';

    if (not os.path.exists(strPath)):
        os.mkdir(strPath);

    files = os.listdir(strPath);

    idx = len(files);

    depth_colormap_image = Image.fromarray(depth_colormap);
    image_path = strPath + str(idx) + '.jpg';
    print(image_path);
    depth_colormap_image.save(image_path);

def removeBackground(image):
    minVal = 10000;

    for x in image:
        for y in x:
            if y < minVal and y != 0:
                minVal = y;

    newImage = [];
    for x in image:
        newRow = [];
        for y in x:
            if y > 1500 + minVal:
                newRow.append(0);
            else:
                newRow.append(y);
        newImage.append(newRow);

    return np.asanyarray(newImage);

# function that gathers camera image data, displays it, and asks for classification
def gatherCameraImage(gesture, iterations):
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Store images to process later so capturing is fast
    storedImages = {};

    print("collecting images");
    for iteration in range(iterations):

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
        # color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())

        #Storing images so we can process later
        key = str(len(storedImages));
        storedImages[key] = depth_image;
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((depth_colormap, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)

    pipeline.stop()

    processedImages = [];

    print("processing images");
    #Processing each stored image
    for image in storedImages:
        processedImages.append(removeBackground(storedImages[image]));

    print("adding images to dataset");
    # Adding each processed image to dataset
    for depth_image in processedImages:
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((depth_colormap, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)

        outputData(depth_colormap,  gesture);

def collectTestingX():
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30);
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30);

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

    depth_image = removeBackground(depth_image);

    depth = frames.get_depth_frame();
    closest = 10000;

    for y in range(480):
        for x in range(640):
            curr = depth.get_distance(x,y);
            closest = curr if curr < closest and curr != 0 else closest;

    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    outputDataToFileStructure(depth_colormap, bg_removed);

    pipeline.stop()

    return (depth_colormap, closest);

def outputDataToFileStructure(depth_image, color_image):
    img = Image.fromarray(color_image, 'RGB')
    img.save("./UI/rgb_image.jpg")
    img = Image.fromarray(depth_image, 'RGB')
    img.save("./UI/depth_image.jpg")

def getNumberLabels():
    dirs = os.listdir('../datasets/');
    return len(dirs);
