import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import wiener
from scipy.fftpack import dct, idct
import cv2
import logging

# implement 2D DCT
def dct2(a):
    return dct(dct(np.float64(a.T), norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(np.float64(a.T), norm='ortho').T, norm='ortho')

def get_filtering_matrix(H, W, c):
    d = np.zeros((H, W), dtype=int)
    d[int(H*c):,int(W*c):] = 1
    return d

def get_hf_noise(img_path, img_width, img_height, plot_results=False):
    with open(img_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint16)

    # Check if the image is read
    if data is None:
        logging.error("Failed to read the image file!")
        return -1
    
    # Reshape the 1D array to a 2D array (image) with the given width and height
    gray_img = data.reshape((img_width, img_height))

    # Check if image is useful
    if gray_img.max() == 0:
        logging.error("The image has no useful data!")
        return -2

    # Filter the image, applying the wiener filter
    filtered_img = wiener(np.float64(gray_img), (5,5))

    # Normalize the image
    filtered_img = np.interp(filtered_img, (filtered_img.min(), filtered_img.max()), (0, 255))

    # Convert the denoised image to float64 data type in order to subtract it to
    # the filtered_img
    gray_img = gray_img.astype(np.float64)

    # Perform the subtraction to obtain the noise
    noise_img = cv2.absdiff(filtered_img, gray_img)
    logging.debug('noise')
    logging.debug(noise_img)

    noise_img = np.interp(noise_img, (noise_img.min(), noise_img.max()), (0, 255))

    # Exctracting high frequencies steps
    # Transform the image in dct domain
    dct_noise = dct2(noise_img)
    logging.debug('dct noise')
    logging.debug(dct_noise)

    # Get the filtering matrix
    (H, W) = gray_img.shape
    c = 0.5
    d = get_filtering_matrix(H, W, c)

    # Filter the noise image, multiplying it with the filtering matrix
    hadamard_prod = np.multiply(dct_noise, d)
    logging.debug('hadamard')
    logging.debug(hadamard_prod)

    # Return to the original domain
    hf_noise = idct2(hadamard_prod)
    hf_noise = np.uint8(np.interp(hf_noise, (hf_noise.min(), hf_noise.max()), (0, 255)))
    logging.debug("hf noise")
    logging.debug(hf_noise)

    # Plotting
    if plot_results:
        f, (plot1, plot2, plot3) = plt.subplots(1, 3)

        plot1.axis("off")
        plot1.set_title("Original")
        plot1.imshow(gray_img)

        plot2.axis("off")
        plot2.set_title("Noise image")
        plot2.imshow(noise_img)

        plot3.axis("off")
        plot3.set_title("HF Noise image")
        plot3.imshow(hf_noise)

        plt.show()

    return hf_noise