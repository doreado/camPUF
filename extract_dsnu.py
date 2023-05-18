import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.datasets import face
from scipy.signal import wiener
from scipy.fftpack import dct, idct

def get_filtering_matrix(H, W, c):
    d = np.zeros((H, W), dtype=int)
    d[int(H*c):,int(W*c):] = 1
    return d

def get_hf_noise(img_path, plot_results=False):
    img = cv2.imread(img_path)

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Filter the image, applying the wiener filter
    filtered_img = wiener(gray_img, (5,5))

    # Normalize the image
    filtered_img = np.interp(filtered_img, (filtered_img.min(), filtered_img.max()), (0, 255))

    # Convert the denoised image to float64 data type in order to subtract it to
    # the filtered_img
    gray_img = gray_img.astype(np.float64)

    # Perform the subtraction to obtain the noise
    noise_img = cv2.absdiff(filtered_img, gray_img)
    noise_img = np.interp(noise_img, (noise_img.min(), noise_img.max()), (0, 255))

    # Exctracting high frequencies steps
    # Transform the image in dct domain
    dct_noise = cv2.dct(np.float32(noise_img))
 
    # Get the filtering matrix
    (H, W) = gray_img.shape
    c = 0.5
    d = get_filtering_matrix(H, W, c)

    # Filter the noise image, multiplying it with the filtering matrix
    hadamard_prod = np.multiply(dct_noise, d)

    # Return to the original domain
    hf_noise = idct(np.float32(hadamard_prod))
    hf_noise = np.interp(hf_noise, (hf_noise.min(), hf_noise.max()), (0, 255))

    if plot_results:
        f, (plot1, plot2, plot3) = plt.subplots(1, 3)

        plot1.axis("off")
        plot1.set_title("Original")
        plot1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plot2.axis("off")
        plot2.set_title("Noise image")
        plot2.imshow(noise_img)

        plot3.axis("off")
        plot3.set_title("HF Noise image")
        plot3.imshow(hf_noise)

        plt.show()

    return hf_noise
