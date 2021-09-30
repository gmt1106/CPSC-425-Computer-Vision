from PIL import Image
import numpy as np
import math
import scipy
from scipy import signal
import cv2


class GaussianFiltering:

    def __init__(self):
        pass

    def boxfilter(self, n):
        # check if the dimension of a filter is odd
        assert (n % 2) != 0, "Dimension must be odd"

        return np.ones((n,n)) * (1/(n**2))


    def gauss1d(self, sigma):
        # length is sigma * 6 rounded up to next odd
        length = np.ceil(sigma * 6)
        if (length % 2) == 0:
            length = length + 1

        length_from_center = np.floor(length/2)

        # the distance of an array value from the center
        x = np.arange(-length_from_center, length_from_center + 1, 1)

        filter = np.exp(-x**2 / (2*(sigma**2)))
        
        # normallize 
        return filter/sum(filter)


    def gauss2d(self, sigma):
        # 1D gaussian filter
        filter1D = self.gauss1d(sigma)

        filter2D = filter1D[np.newaxis]
        transpose_of_filter2D = np.transpose(filter2D)
    
        return scipy.signal.convolve2d(filter2D, transpose_of_filter2D)


    def convolve2d_manual(self, array, filter):
        image_height, image_width = np.shape(array)
        filter_dimension, temp = np.shape(filter)

        # find out how many 0 padding row and column should be added to the image
        zero_padding_size = int(np.floor(filter_dimension / 2))

        # add zero padding to the image
        padded_image_array = np.pad(array,((zero_padding_size, zero_padding_size),(zero_padding_size, zero_padding_size)), mode='constant')
        padded_image_height, padded_image_width = np.shape(padded_image_array)

        # convolued image initialization
        convoluted_image_array = np.zeros((image_height, image_width))

        # manual convolution
        for i in range(image_height):
            for j in range(image_width):
                convoluted_image_array[i][j] = np.sum(np.rot90(filter, 2) * padded_image_array[i : i + filter_dimension, j : j + filter_dimension])

        return convoluted_image_array


    def gaussconvolve2d_manual(self, array, sigma):
        filter = self.gauss2d(sigma)
        return self.convolve2d_manual(array, filter)

    
    def gaussconvolve2d_scipy(self, array, sigma):
        filter = self.gauss2d(sigma)
        return signal.convolve2d(array, filter, 'same')

    ##### this function is written for test purpose of quesiton 5 part (a) #####
    def gausscorrelate2d_scipy(self, array, sigma):
        filter = self.gauss2d(sigma)
        return signal.correlate2d(array,filter,'same')




