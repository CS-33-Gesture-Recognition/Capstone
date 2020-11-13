import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as color
import pyrealsense2 as rs
import numpy as np
import time
#img = mpimg.imread('test.png')
#imgplot = plt.imshow(img)
#plt.show()

plt.ion()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

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

def removeBackground(image):
    minVal = np.min(image[np.nonzero(image)])

    image[image > 1500 + minVal] = 0

    return image

def rescale(image):
    #minVal = np.min(image[np.nonzero(image)])
    maxVal = np.max(image[np.nonzero(image)])

    image[image == 0] = maxVal

    # lets aim for a range of 0 -> 1000
    new_img = (image / maxVal) ** 4

    return new_img

def get_frame():
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image

    depth_image = np.asanyarray(aligned_depth_frame.get_data())

    return removeBackground(depth_image)


while(True):

    img = get_frame()
    img = rescale(img)
    #print("min: " + str(np.min(img[np.nonzero(img)])) + " max: " + str(np.max(img[np.nonzero(img)])))
    imgplot = plt.imshow(img, "plasma")
    plt.show()
    plt.pause(0.001)
    x = input("?")
    if (x != ''):
        break
