import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import rawpy
from scipy.signal import wiener
from scipy.fftpack import dct, idct
import rawpy
import cv2

def read_cr2_image(cr2_file_path):
    with rawpy.imread(cr2_file_path) as raw:
        rgb_image = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)
        return rgb_image

def denoise_cr2_image(cr2_image_path):
    # Convert image to 8-bit unsigned integer format (CV_8U)
    # cr2_image_uint8 = (cr2_image * 255).astype(np.uint8)
    cr2_image = read_cr2_image(cr2_image_path)
    cr2_image_uint8 = cr2_image.astype(np.uint8)

    # Apply Non-Local Means Denoising
    denoised_image_uint8 = cv2.fastNlMeansDenoisingColored(cr2_image_uint8, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Convert back to floating-point format
    # denoised_image = denoised_image_uint8.astype(np.float32) / 255.0
    denoised_image = denoised_image_uint8

    return denoised_image

# TODO test it
def wiener_filter(image, nsr):
    # Compute the power spectral density of the observed image
    psd_image = np.abs(np.fft.fft2(image)) ** 2

    # Compute the Wiener filter
    wiener_filter = 1 / (1 + nsr / psd_image)

    # Apply the Wiener filter in the frequency domain
    filtered_image = np.fft.ifft2(np.fft.fft2(image) * wiener_filter).real

    # Clip the pixel values to the valid range [0, 1]
    filtered_image = np.clip(filtered_image, 0, 1)

    return filtered_image

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

def get_hf_noise(img_path, plot_results=False):
    # img = cv2.imread(img_path)
    # img = rawpy.imread(img_path)
    # img = img.postprocess()
    
    # Check if the image is read
    if img_path is None:
        print("Failed to read the image file!")
        return -1

    # Convert image to grayscale
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = read_cr2_image(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Check if image is useful
    # print(np.max(gray_img))
    # if gray_img.max() == 0:
    #     print("The image has no useful data!")
    #     return -2

    # Filter the image, applying the wiener filter
    # filtered_img = wiener(np.float64(gray_img), (5,5))
    # filtered_img = wiener(np.float64(img), (5,5))
    filtered_img = denoise_cr2_image(img_path)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)

    # Normalize the image
    # filtered_img = np.interp(filtered_img, (filtered_img.min(), filtered_img.max()), (0, 255))

    # Convert the denoised image to float64 data type in order to subtract it to
    # the filtered_img
    # gray_img = gray_img.astype(np.float64)

    # Perform the subtraction to obtain the noise
    noise_img = cv2.absdiff(filtered_img, img)
    print('noise')
    print(noise_img)

    # noise_img = np.interp(noise_img, (noise_img.min(), noise_img.max()), (0, 255))

    # Exctracting high frequencies steps
    # Transform the image in dct domain
    # dct_noise = dct(np.float64(noise_img), cv2.DCT_INVERSE)
    # dct_noise = dct(np.float64(noise_img), cv2.DCT_INVERSE)
    dct_noise = dct2(noise_img)
    print('dct noise')
    print(dct_noise)

    # Get the filtering matrix
    (H, W) = img.shape
    c = 0.5
    d = get_filtering_matrix(H, W, c)

    # Filter the noise image, multiplying it with the filtering matrix
    hadamard_prod = np.multiply(dct_noise, d)
    print('hadamard')
    print(hadamard_prod)

    # Return to the original domain
    hf_noise = idct2(hadamard_prod)
    hf_noise = np.uint8(np.interp(hf_noise, (hf_noise.min(), hf_noise.max()), (0, 255)))
    print("hf noise")
    print(hf_noise)

    if plot_results:
        f, (plot1, plot2, plot3) = plt.subplots(1, 3)

        plot1.axis("off")
        plot1.set_title("Original")
        plot1.imshow(img)

        plot2.axis("off")
        plot2.set_title("Noise image")
        plot2.imshow(noise_img)

        plot3.axis("off")
        plot3.set_title("HF Noise image")
        plot3.imshow(hf_noise)

        plt.show()

    return hf_noise

# TEST
# script_dir = os.path.dirname(os.path.abspath(__file__))
# img_path_v = os.path.join(script_dir, 'downloads', 'vincenzo_t.DNG')
# img_path_a = os.path.join(script_dir, 'downloads', 'andrea_t.DNG')
# img_path_g = os.path.join(script_dir, 'downloads', 'giulia_t.DNG')
# img_path_gB = os.path.join(script_dir, 'downloads', 'giulia_tas.DNG')
# img_path_m = os.path.join(script_dir, 'downloads', 'marco_t.DNG')
#
# img_path_test = './images/pictures/2_zaino.DNG'
# img_path_test = './images/IMG_5985.CR2'
# img_path_test = './images/IMG_5985.CR2'
# get_hf_noise(img_path_test, True)

#img_path_test = './images/pictures/1_tasca.DNG'
# img_path_test = './images/IMG_5986.CR2'
# get_hf_noise(img_path_test, True)
