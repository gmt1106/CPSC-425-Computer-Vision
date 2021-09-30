from PIL import Image
import numpy as np
import math
import scipy
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import argparse
import time

from gaussian_filtering import GaussianFiltering


if __name__ == "__main__":

    ###### how to run this program ########

    # python main.py -q 'quesiton_number'
    # queiston_number is one of ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3", "4.1", "4.2"]

    #######################################


    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "3", "4.1", "4.2"])

    args = parser.parse_args()
    question = args.question

    if question == "2.1":
        # Show the results of your boxfilter(n) function for the cases n=3, n=4, and n=5.

        filter_class = GaussianFiltering()
        print("n = 3")
        print(filter_class.boxfilter(3))
        print("n = 5")
        print(filter_class.boxfilter(5))
        print("n = 4")
        print(filter_class.boxfilter(4))

    if question == "2.2":
        # Show the filter values produced for sigma values of 0.3, 0.5, 1, and 2.

        filter_class = GaussianFiltering()
        print("sigma = 0.3")
        print(filter_class.gauss1d(0.3))
        print("sigma = 0.5")
        print(filter_class.gauss1d(0.5))
        print("sigma = 1")
        print(filter_class.gauss1d(1))
        print("sigma = 2")
        print(filter_class.gauss1d(2))

    if question == "2.3":
        # Show the 2D Gaussian filter for sigma values of 0.5 and 1.

        filter_class = GaussianFiltering()
        print("sigma = 0.5")
        print(filter_class.gauss2d(0.5))
        print("sigma = 1")
        print(filter_class.gauss2d(1))

    if question == "2.4":
        # Apply your ‘gaussconvolve2d_manual’ with a sigma of 3 on an image

        image = Image.open('./images/dog.jpeg')

        # turn the iamge to greyscale image
        greyscaled_image = image.convert('L')

        # turn greyscle image to array
        image_array = np.asarray(greyscaled_image)

        filter_class = GaussianFiltering()

        convolued_image_array = filter_class.gaussconvolve2d_manual(image_array, 3)

        # convert the array back to a unit8 array so we can write to a file
        convolued_image_array = convolued_image_array.astype('uint8')
        convolued_image = Image.fromarray(convolued_image_array)

        # save the blurred image 
        convolued_image.save('./results/blurred_dog_gaussconvolve2d_manual.png','PNG')

        # display the original and blurred image
        convolued_image.show()
        image.show()

    if question == "2.5":
        #  Apply your ‘gaussconvolve2d_scipy’ with a sigma of 3 on an image

        image = Image.open('./images/dog.jpeg')

        # turn the iamge to greyscale image
        greyscaled_image = image.convert('L')

        # turn greyscle image to array
        image_array = np.asarray(greyscaled_image)

        filter_class = GaussianFiltering()

        ### spicy convolution ###
        convolued_image_array = filter_class.gaussconvolve2d_scipy(image_array, 3)

        # convert the array back to a unit8 array so we can write to a file
        convolued_image_array = convolued_image_array.astype('uint8')
        convolued_image = Image.fromarray(convolued_image_array)

        # save the blurred image 
        convolued_image.save('./results/blurred_dog_gaussconvolve2d_scipy.png','PNG')

        # display the original and blurred image
        convolued_image.show()
        image.show()

        #### spicy correlation for test ####
        # written for test purpose of quesiton 5 part (a)
        # correlated_image_array = filter_class.gausscorrelate2d_scipy(image_array, 3)

        # # convert the array back to a unit8 array so we can write to a file
        # correlated_image_array = correlated_image_array.astype('uint8')
        # correlated_image = Image.fromarray(correlated_image_array)

        # # save the blurred image 
        # correlated_image.save('blur_dog_by_signal_correlation.png','PNG')

        # # display the blurred image
        # correlated_image.show()


    if question == "2.6":
        # Experiment on how much time it takes to convolve the dog image above using your convolution implementation ‘gaussconvolve2d_manual’ and the scipy implementation ‘gaussconvolve2d’

        image = Image.open('./images/dog.jpeg')

        # turn the iamge to greyscale image
        greyscaled_image = image.convert('L')

        # turn greyscle image to array
        image_array = np.asarray(greyscaled_image)

        filter_class = GaussianFiltering()


        ### manual convolution ###
        start_time = time.time()
        convolued_image_array = filter_class.gaussconvolve2d_manual(image_array, 10)
        duration = time.time() - start_time
        print("manual convolution took:" + str(duration))


        ### spicy convolution ###
        start_time = time.time()
        convolued_image_array = filter_class.gaussconvolve2d_scipy(image_array, 10)
        duration = time.time() - start_time
        print("spicy convolution took:" + str(duration))


    if question == "3":
        # A hybrid image is the sum of a low-pass filtered version of the one image and a high-pass filtered version of a second image.
        # Make hybrid images with 3 different sets of each for 3 different sigma value. 

        image_sets = [["./images/1b_motorcycle.bmp", "./images/1a_bicycle.bmp"], ["./images/2b_marilyn.bmp", "./images/2a_einstein.bmp"], ["./images/0b_dog.bmp", "./images/0a_cat.bmp"]]
        sigma = [3, 7, 11]

        for i in range(3):
            for j in sigma:
                ####### 3.1 #######
                image_a = Image.open(image_sets[i][0])

                image_a_array = np.asarray(image_a)
                height, width, channel = np.shape(image_a_array)

                # seperate the image by 3 color channels
                image_a_array_blue, image_a_array_green, image_a_array_red = image_a_array[:,:,0], image_a_array[:,:,1], image_a_array[:,:,2]

                filter_class = GaussianFiltering()

                #spicy convolution to all 3 color channels
                low_frequency_image_array = np.zeros((height, width, channel))
                low_frequency_image_array[:,:,0] = filter_class.gaussconvolve2d_scipy(image_a_array_blue, j)
                low_frequency_image_array[:,:,1]= filter_class.gaussconvolve2d_scipy(image_a_array_green, j)
                low_frequency_image_array[:,:,2] = filter_class.gaussconvolve2d_scipy(image_a_array_red, j)

                # clamping the values of pixels on the high and low end to ensure they are in the valid range (between 0 and 255)
                low_frequency_image_array[low_frequency_image_array > 255] = 255
                low_frequency_image_array[low_frequency_image_array < 0] = 0

                # convert the array back to a unit8 array so we can write to a file
                low_frequency_image_array_visualization = low_frequency_image_array.astype('uint8')
                low_frequency_image_visualization = Image.fromarray(low_frequency_image_array_visualization)

                # save the blurred image 
                low_frequency_image_visualization.save('./results/low_frequencies_image_' + str(i) + '_sigma_' + str(j) + '.png','PNG')


                ####### 3.2 #######
                image_b = Image.open(image_sets[i][1])

                image_b_array = np.asarray(image_b)
                height, width, channel = np.shape(image_b_array)

                # seperate the image by 3 color channels
                image_b_array_blue, image_b_array_green, image_b_array_red = image_b_array[:,:,0], image_b_array[:,:,1], image_b_array[:,:,2]

                filter_class = GaussianFiltering()

                #spicy convolution to all 3 color channels
                low_frequency_image_b_array = np.zeros((height, width, channel))
                low_frequency_image_b_array[:,:,0] = filter_class.gaussconvolve2d_scipy(image_b_array_blue, j)
                low_frequency_image_b_array[:,:,1]= filter_class.gaussconvolve2d_scipy(image_b_array_green, j)
                low_frequency_image_b_array[:,:,2] = filter_class.gaussconvolve2d_scipy(image_b_array_red, j)

                # produce high frequency filtered image by subtracting low frequency Gaussian filtered image from the original
                high_frequency_image_array = image_b_array - low_frequency_image_b_array

                # convert the array back to a unit8 array so we can write to a file
                high_frequency_image_array_visualization = high_frequency_image_array + 128

                # clamping the values of pixels on the high and low end to ensure they are in the valid range (between 0 and 255)
                high_frequency_image_array_visualization[high_frequency_image_array_visualization > 255] = 255
                high_frequency_image_array_visualization[high_frequency_image_array_visualization < 0] = 0

                high_frequency_image_array_visualization = high_frequency_image_array_visualization.astype('uint8')
                high_frequency_image_visualization = Image.fromarray(high_frequency_image_array_visualization)

                # save the blurred image 
                high_frequency_image_visualization.save('./results/high_frequencies_image_' + str(i) + '_sigma_' + str(j) + '.png','PNG')


                ####### 3.3 #######
                hybrid_image_array = high_frequency_image_array + low_frequency_image_array

                # clamping the values of pixels on the high and low end to ensure they are in the valid range (between 0 and 255)
                hybrid_image_array[hybrid_image_array > 255] = 255
                hybrid_image_array[hybrid_image_array < 0] = 0

                # convert the array back to a unit8 array so we can write to a file
                hybrid_image_array = hybrid_image_array.astype('uint8')
                hybrid_image = Image.fromarray(hybrid_image_array)

                # save the blurred image 
                hybrid_image.save('./results/hybrid_image_' + str(i) + '_sigma_' + str(j) + '.png','PNG')


    if question == "4.1":
        # Fiven two images affected by Gaussian noise and speckle noise, apply Gaussian filter, bilateral filter, and median filter respectively to denoise the images.
        # Use the functions ‘cv2.GaussianBlur’, ‘cv2.bilateralFilter’, and ‘cv2.medianBlur’.

        # read image
        box_gauss_image = cv2.imread('./images/box_gauss.png', cv2.IMREAD_UNCHANGED)
        box_speckle_image = cv2.imread('./images/box_speckle.png', cv2.IMREAD_UNCHANGED)

        ###### test run code for finding the best paramerter is commented #######

        # filter_dimension = [3, 5, 7, 9]
        # sigma = [0.2, 0.5, 0.7, 1, 1.5, 2]

        # for i in filter_dimension:
        #     for j in sigma:
        #         # apply guassian blur on image
        #         guassian_blur_box_gauss_image = cv2.GaussianBlur(box_gauss_image, ksize=(i,i), sigmaX=j, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
        #         guassian_blur_box_speckle_image = cv2.GaussianBlur(box_speckle_image,ksize=(i,i), sigmaX=j, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

        #         # save image
        #         cv2.imwrite('guassian_blur_box_gauss_image_ksize_' + str(i) +'_sigma_' + str(j) + '.png', guassian_blur_box_gauss_image)
        #         cv2.imwrite('guassian_blur_box_speckle_image_ksize_' + str(i) +'_sigma_' + str(j) + '.png', guassian_blur_box_speckle_image)


        # d = [5, 10, 15, 20]
        # sigma = [70, 80, 90, 100]

        # for i in d:
        #     for j in sigma:
        #         # apply Bilateral Filter on image
        #         bilateral_filter_box_gauss_image = cv2.bilateralFilter(box_gauss_image, d=i, sigmaColor=j, sigmaSpace=j, borderType=cv2.BORDER_DEFAULT)
        #         bilateral_filter_box_speckle_image = cv2.bilateralFilter(box_speckle_image,d=i, sigmaColor=j, sigmaSpace=j, borderType=cv2.BORDER_DEFAULT)

        #         # save image
        #         cv2.imwrite('bilateral_filter_box_gauss_image_d_' + str(i) +'_sigma_' + str(j) + '.png', bilateral_filter_box_gauss_image)
        #         cv2.imwrite('bilateral_filter_box_speckle_image_d_' + str(i) +'_sigma_' + str(j) + '.png', bilateral_filter_box_speckle_image)


        # filter_dimension = [3, 5, 7, 9]

        # for i in filter_dimension:
        #     # apply Bilateral Filter on image
        #     median_blur_box_gauss_image = cv2.medianBlur(box_gauss_image, ksize=i)
        #     median_blur_box_speckle_image = cv2.medianBlur(box_speckle_image, ksize=i)

        #     # save image
        #     cv2.imwrite('median_blur_box_gauss_image_ksize_' + str(i) + '.png', median_blur_box_gauss_image)
        #     cv2.imwrite('median_blur_box_speckle_image_ksize_' + str(i) + '.png', median_blur_box_speckle_image)


        #### best result ####

        #apply guassian blur on image
        guassian_blur_box_gauss_image = cv2.GaussianBlur(box_gauss_image, ksize=(7,7), sigmaX=1, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
        guassian_blur_box_speckle_image = cv2.GaussianBlur(box_speckle_image,ksize=(9,9), sigmaX=1, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

        # save image
        cv2.imwrite('./results/guassian_blur_box_gauss_image_ksize_' + str(7) +'_sigma_' + str(1) + '.png', guassian_blur_box_gauss_image)
        cv2.imwrite('./results/guassian_blur_box_speckle_image_ksize_' + str(9) +'_sigma_' + str(1) + '.png', guassian_blur_box_speckle_image)

        # apply Bilateral Filter on image
        bilateral_filter_box_gauss_image = cv2.bilateralFilter(box_gauss_image, d=10, sigmaColor=80, sigmaSpace=80, borderType=cv2.BORDER_DEFAULT)
        bilateral_filter_box_speckle_image = cv2.bilateralFilter(box_speckle_image,d=10, sigmaColor=80, sigmaSpace=80, borderType=cv2.BORDER_DEFAULT)

        # save image
        cv2.imwrite('./results/bilateral_filter_box_gauss_image_d_' + str(10) +'_sigma_' + str(80) + '.png', bilateral_filter_box_gauss_image)
        cv2.imwrite('./results/bilateral_filter_box_speckle_image_d_' + str(10) +'_sigma_' + str(80) + '.png', bilateral_filter_box_speckle_image)

        # apply Bilateral Filter on image
        median_blur_box_gauss_image = cv2.medianBlur(box_gauss_image, ksize=5)
        median_blur_box_speckle_image = cv2.medianBlur(box_speckle_image, ksize=5)

        # save image
        cv2.imwrite('./results/median_blur_box_gauss_image_ksize_' + str(5) + '.png', median_blur_box_gauss_image)
        cv2.imwrite('./results/median_blur_box_speckle_image_ksize_' + str(5) + '.png', median_blur_box_speckle_image)


    if question == "4.2":
        # Also denosing as 4.1, but use specific parameters that are given. 

         # read image
        box_gauss_image = cv2.imread('./images/box_gauss.png', cv2.IMREAD_UNCHANGED)
        box_speckle_image = cv2.imread('./images/box_speckle.png', cv2.IMREAD_UNCHANGED)

        guassian_blur_box_gauss_image = cv2.GaussianBlur(box_gauss_image, ksize=(7, 7), sigmaX=50)
        bilateral_filter_box_gauss_image = cv2.bilateralFilter(box_gauss_image, 7, sigmaColor=150, sigmaSpace=150)
        median_blur_box_gauss_image = cv2.medianBlur(box_gauss_image,7)

        cv2.imwrite('./results/guassian_blur_box_gauss_image.png', guassian_blur_box_gauss_image)
        cv2.imwrite('./results/bilateral_filter_box_gauss_image.png', bilateral_filter_box_gauss_image)
        cv2.imwrite('./results/median_blur_box_gauss_image.png', median_blur_box_gauss_image)

        guassian_blur_box_speckle_image = cv2.GaussianBlur(box_speckle_image, ksize=(7, 7), sigmaX=50)
        bilateral_filter_box_speckle_image = cv2.bilateralFilter(box_speckle_image, 7, sigmaColor=150, sigmaSpace=150)
        median_blur_box_speckle_image = cv2.medianBlur(box_speckle_image,7)

        cv2.imwrite('./results/guassian_blur_box_speckle_image.png', guassian_blur_box_speckle_image)
        cv2.imwrite('./results/bilateral_filter_box_speckle_image.png', bilateral_filter_box_speckle_image)
        cv2.imwrite('./results/median_blur_box_speckle_image.png', median_blur_box_speckle_image)