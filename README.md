# Hybrid Images
## Introduction 
This project aims to develop various filters to create new images from the original ones. Image filtering is a fundamental tool in image processing, and this project focuses on implementing several key types of filters and techniques:
1. Gaussian Blur: Will blur the image by taking an unweighted average of the neighboring pixels.
2. Sharpening Filter: Enhances image details and edges to make them more defined.
3. Morphological Operations: Improves image quality and structure.
4. High and Low Pass Filters: Will remove fine details (low pass) or retails fine details (high
pass).
5. Cross-Correlation and Convolution: Operations for image filtering.
6. Multi-Scale Hybrid Images: Combines the low pass part of one image with the high pass part
of another to create images that look different based on the viewing distance.
These filters and techniques will be used to create hybrid images, which will be affected considering the visual distance.

## Detailed explanation of each method

1. cross_correlation_2D(image, kernel):
• This function performs a 2D cross-correlation operation between an image and a given kernel.
• It first calculates the height and width of the image and the kernel, and determines the padding needed to apply the kernel to the image.
• It then creates a padded image with zeros around the original image to handle the border cases.
• Next, it initializes an output image with the same dimensions as the original image.
• Finally, it applies the kernel to the padded image by sliding the kernel over the image, performing the element-wise multiplication between the kernel and the corresponding
region of the padded image, and summing the results to obtain the final output pixel value.

2. convolve_2d(image, kernel):
• This function applies 2D convolution to an image using a given kernel.
• It first retrieves the dimensions of the kernel.
• It then creates a flipped version of the kernel by reversing the order of the rows and
columns.
• Finally, it calls the cross_correlation_2d function with the image and the flipped kernel
to perform the convolution.

3. gaussian_blur_kernel_2d(size, sigma):
• This function creates a 2D Gaussian blur kernel of a specified size and standard deviation (sigma).
• It first initializes an empty kernel and calculates the center of the kernel.
• It then fills the kernel by iterating over each element and computing the Gaussian value
based on the distance from the center.
• Finally, it normalizes the kernel by dividing each element by the sum of all the
elements, ensuring that the kernel values sum up to 1.

4. gaussian_blur(image, size, sigma):
• This function applies Gaussian blur to an image.
• It first creates a Gaussian blur kernel using the gaussian_blur_kernel_2d function.
• Then, it calls the convolve_2d function to apply the Gaussian kernel to the input image,
effectively performing the Gaussian blur.

5. gradient_filter(image):
• This function applies a sharpening filter to an image using a predefined kernel.
• It defines a 3x3 sharpening kernel, which is a standard kernel for sharpening taken from
Wikipedia.
• It then calls the convolve_2d function to apply the sharpening kernel to the input image,
resulting in a sharpened image.

6. morph_op(image, operation):
• This function performs morphological operations (dilation or erosion) on an image.
• It defines a 3x3 kernel filled with 1's.
• If the operation parameter is 'dilation', it calls the cross_correlation_2d function to
apply dilation to the input image.
• If the operation parameter is 'erosion', it calls the convolve_2d function to apply erosion
to the input image.

7. high_pass(image, size, sigma):
• This function applies a high-pass filter to an image.
• It first calls the gaussian_blur function to apply Gaussian blur to the input image,
effectively creating a low-pass filtered version.
• Then, it subtracts the blurred image from the original image to extract the high-
frequency details, resulting in the high-pass filtered image.

8. low_pass(image, size, sigma):
• This function applies a low-pass filter to an image.
• It simply calls the gaussian_blur function with the specified parameters to apply
Gaussian blur to the input image, effectively creating a low-pass filtered version.

9. hybrid_image(img1, img2, size1, sigma1, size2, sigma2):
• This function creates a hybrid image by combining a low-pass filtered version of one image with a high-pass filtered version of another image.
• It first calls the low_pass function to apply a low-pass filter to the first input image (img1) using the specified parameters (size1, sigma1).
• It then calls the high_pass function to apply a high-pass filter to the second input image (img2) using the specified parameters (size2, sigma2).
• To ensure that both filtered images have the same dimensions, it finds the minimum height and width between the two filtered images.
• Finally, it combines the low-pass and high-pass filtered images by adding the corresponding pixel values, creating the hybrid image.

10. load_image(path):
• This function loads an image from the specified path and converts it to grayscale.
• It uses the Image.open and Image.convert functions from the PIL (Python Imaging
Library) to load and convert the image.
• It then extracts the pixel values from the image and stores them in a 2D list, which is
returned as the output.

11. show_image(image, title=''):
• This function displays the input image using the matplotlib.pyplot library.
• It applies grayscale colormap to the image and sets the title of the plot.
• It also turns off the axis display to show the image only.
