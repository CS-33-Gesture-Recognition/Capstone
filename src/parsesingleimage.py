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


# function used to normalize data from all real numbers to [0,1]
def normalizeDepthImage(depth_image):

    largest = [];
    depth_image_normalized = [];

    for x in range(len(depth_image)):
        largest.append(0.0);
        for i in range(len(depth_image[x])):
            if (float(depth_image[x][i]) > largest[x]):
                largest[x] = float(depth_image[x][i]);
        
    for x in range(len(depth_image)):
        temp = [];
        for i in range(len(depth_image[x])):
            if (largest[x] != 0.0):
                temp.append(float(depth_image[x][i]) / float(largest[x]));
                depth_image[x][i] = temp[i];
        depth_image_normalized.append(temp);

    return np.asanyarray(depth_image_normalized);

def outputData(outputdepthtotal, outputdepth, depth_image):
    print("Outputting Depth Image\n")
    print("Size of depth image: " + str(depth_image.size) + " \n")
    outputdepth.write("[");
    outputdepthtotal.write("[");
    for i in range(len(depth_image)):
        outputdepth.write("[");
        outputdepthtotal.write("[");
        for x in range(len(depth_image[i])):
            if (x == (len(depth_image[i]) - 1)):
                outputdepth.write(str(depth_image[i][x]));
                outputdepthtotal.write(str(depth_image[i][x]));
            else:
                outputdepth.write(str(depth_image[i][x]) + ",");
                outputdepthtotal.write(str(depth_image[i][x]) + ",");
        if (i == (len(depth_image) - 1)):
            outputdepth.write("]");
            outputdepthtotal.write("]");
        else:
            outputdepth.write("],");
            outputdepthtotal.write("],");
    outputdepth.write("]");
    outputdepthtotal.write("],");

def gather_camera_image():
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
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    outputdepthtotal = open('imagedepth', 'a')
    outputdepth = open('singleimagedepth', 'w+')

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
    
    #Normalize Depth Image
    depth_image = normalizeDepthImage(depth_image);

    #Writing depth image to file
    outputData(outputdepthtotal, outputdepth, depth_image);

    pipeline.stop()