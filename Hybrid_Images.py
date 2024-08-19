import matplotlib.pyplot as plt
from PIL import Image
import math


# Cross correlation on a 2d image using a given kernel
def cross_correlation_2d(image, kernel):
    # Store height and width of image and kernel, calculate the padding needed to apply for the image
    img_height = len(image)
    img_width = len(image[0])
    ker_height = len(kernel)
    ker_width = len(kernel[0])
    pad_height = ker_height // 2
    pad_width = ker_width // 2

    # Make padded image with zeros
    padded_image = []
    for i in range(img_height + 2 * pad_height):
        row = [0] * (img_width + 2 * pad_width)
        padded_image.append(row)
    # Copy the original image into the center of the padded image
    for i in range(img_height):
        for j in range(img_width):
            padded_image[i + pad_height][j + pad_width] = image[i][j]

    # Create output image
    output = []
    for i in range(img_height):
        row = [0] * img_width
        output.append(row)
    # Apply the kernel to the padded image
    for i in range(img_height):
        for j in range(img_width):
            sum = 0
            for m in range(ker_height):
                for n in range(ker_width):
                    sum += padded_image[i + m][j + n] * kernel[m][n]
            output[i][j] = sum

    return output


# Method to perform convolution by flipping the kernel and applying cross-correlation
def convolve_2d(image, kernel):
    # Get dimensions of the kernel
    ker_height = len(kernel)
    ker_width = len(kernel[0])
    # Flip the kernel horizontally and vertically
    flipped_kernel = []
    for i in range(ker_height):
        row = []
        for j in range(ker_width):
            row.append(kernel[ker_height - 1 - i][ker_width - 1 - j])
        flipped_kernel.append(row)
    # Apply cross-correlation with the flipped kernel
    return cross_correlation_2d(image, flipped_kernel)


# Method to create a Gaussian blur kernel
def gaussian_blur_kernel_2d(size, sigma):

    kernel = []
    center = size // 2
    total = 0
    # Fill the kernel
    for i in range(size):
        row = []
        for j in range(size):
            x = i - center
            y = j - center
            value = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            row.append(value)
            total += value
        kernel.append(row)
    # Normalize the kernel
    for i in range(size):
        for j in range(size):
            kernel[i][j] = kernel[i][j] / total

    return kernel


# Method to apply Gaussian blur to an image
def gaussian_blur(image, size, sigma):
    # Create a Gaussian blur kernel
    kernel = gaussian_blur_kernel_2d(size, sigma)
    # Apply convolution with the Gaussian kernel
    return convolve_2d(image, kernel)


# Method to apply a sharpening filter to an image
def gradient_filter(image):
    # Define a sharpening kernel (Standard kernel for sharpening taken from Wikipedia)
    kernel = [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]
    # Apply convolution with the sharpening kernel
    return convolve_2d(image, kernel)


# Method to perform morphological operations on an image
def morph_op(image, operation):
    # Define a 3x3 kernel filled with 1's
    kernel = []
    for _ in range(3):
        row = [1] * 3
        kernel.append(row)

    if operation == 'dilation':
        # Apply dilation using cross-correlation
        return cross_correlation_2d(image, kernel)
    elif operation == 'erosion':
        # Apply erosion using convolution
        return convolve_2d(image, kernel)


# Method to apply a high-pass filter to an image
def high_pass(image, size, sigma):
    # Blur the image using gaussian blur
    blurred = gaussian_blur(image, size, sigma)
    high_pass_image = []
    for i in range(len(image)):
        row = []
        for j in range(len(image[0])):
            # Subtract the blurred image from the original image to get high-frequency details
            row.append(image[i][j] - blurred[i][j])
        high_pass_image.append(row)
    return high_pass_image


# Method to apply a low-pass filter to an image
def low_pass(image, size, sigma):
    # Apply Gaussian blur to get a low-pass filter
    return gaussian_blur(image, size, sigma)


# Method to create a hybrid image from 2 images
def hybrid_image(img1, img2, size1, sigma1, size2, sigma2):
    # Apply low-pass filter to the first image
    low_pass_img = low_pass(img1, size1, sigma1)
    # Apply high-pass filter to the second image
    high_pass_img = high_pass(img2, size2, sigma2)
    # Ensure both images have the same dimensions
    min_height = min(len(low_pass_img), len(high_pass_img))
    min_width = min(len(low_pass_img[0]), len(high_pass_img[0]))
    # Combine low-pass and high-pass images to create the hybrid image
    hybrid_img = []
    for i in range(min_height):
        row = []
        for j in range(min_width):
            row.append(low_pass_img[i][j] + high_pass_img[i][j])
        hybrid_img.append(row)

    return hybrid_img


def load_image(path):
    # 'L' means converting the image to gray scale
    image = Image.open(path).convert('L')
    img_width = image.width
    img_height = image.height
    pixels = []

    for i in range(img_height):
        row = []
        for j in range(img_width):
            row.append(image.getpixel((j, i)))
        pixels.append(row)

    return pixels


# Method to show the image and convert to gray scale
def show_image(image, title=''):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')


if __name__ == "__main__":

    img1 = load_image('motorcycle.bmp')
    img2 = load_image('bicycle.bmp')
    size1 = 15
    sigma1 = 3
    size2 = 15
    sigma2 = 3

    # Create the hybrid image
    hybrid = hybrid_image(img1, img2, size1, sigma1, size2, sigma2)

    # Display the result
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    show_image(img1, title='Image 1')

    plt.subplot(1, 3, 2)
    show_image(img2, title='Image 2')

    plt.subplot(1, 3, 3)
    show_image(hybrid, title='Hybrid Image')

    plt.tight_layout()
    plt.show()

