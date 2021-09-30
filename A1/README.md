# Image Filtering and Hybrid Images 

### Implementation in gaussian_filtering.py
+ boxfilter(n): returns n by n boxfilter
+ gauss1d(sigma): returns a 1D Gaussian filter for a given value of sigma
+ gauss2d(sigma): returns a 2D Gaussian filter for a given value of sigma
+ convolve2d_manual(array, filter): takes in an image (stored in array) and a filter, and performs convolution to the image with zero paddings 
+ gaussconvolve2d_manual(array,sigma): applies Gaussian convolution to a 2D array for the given value of sigma
+ gaussconvolve2d_scipy(array,sigma): applies Gaussian convolution to a 2D array for the given value of sigma using signal.convolve2d(array,filter,'same')


### How to run the code 
It can be run by  `python main.py -q ‘question_number’` and question_number is one of  [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3, 4.1, 4.2].
For example, “2.1” means part 2, question 1 in the assignment. 
