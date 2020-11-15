import numpy as np
# import RSC_Wrapper as RSCW


class PreProc:
    def __init__(self):
        print("Hello!")

    def fix_scale(self, image):
        # First, find the minimum value in the depth image
        # for the depth image, the closer the object is to the camera,
        #  the smaller
        # the value. Thus, we want to find the closest pixel
        minVal = np.min(image[np.nonzero(image)])

        # Now we want to mask off the background. We do this by
        # setting any pixel thats further away than 1500 units from
        # the closest pixel to zero
        image[image > 1500 + minVal] = 0.

        # After masking off the background, find the furthest distance
        # in our ROI
        # (note, this could be minVal + 1500, but it could be smaller)
        maxVal = np.max(image[np.nonzero(image)])

        # Now we preform two operations at once, the first is to scale the ROI
        # relitive to itself
        # as such, the closest pixel should be near zero, and the furthest
        # near one.
        # Second, we raise this value to the fourth power, this is to help
        #  make minor differences
        # between pixels more distinct.
        # For example: the sign 'A' versus the sign 'S'
        # 'A' has the thumb closer to the camera than in 'S', but the
        # difference is thousanths of units. Thus, by raising to the fourth
        #  power, we can increase that difference
        new_img = (image / maxVal) ** 4

        # Any small value (arbitrarily defined as smaller than 0.001,
        #  typically in the range ~E-5) is the
        # result of floating point errors with the masked off pixels
        # (I beleive)
        # Set these background pizels as 1, the furthest away in our range
        new_img[new_img < 0.001] = 1

        return new_img
