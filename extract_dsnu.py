import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.datasets import face
from scipy.signal import wiener
from scipy.fftpack import dct, idct

# https://stackoverflow.com/questions/7110899/how-do-i-apply-a-dct-to-an-image-in-python
# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    


def get_filtering_matrix(H, W, c):
    d = np.zeros((H, W), dtype=int)
    d[int(H*c):,int(W*c):] = 1
    return d

img = cv2.imread("./images/IMG_5985.CR2")

# Convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filter the image, applying the wiener filter
filtered_img = wiener(gray_img, (5,5))

# Convert the denoised image to uint8 data type in order to subtract it to the
# filtered_img
gray_img = gray_img.astype(np.float64)

# Perform the subtraction to obtain the noise
noise_img = cv2.absdiff(gray_img, filtered_img)

# Exctracting high frequencies steps
# Transform the image in dct domain
dct2(noise_img)

# Get the filtering matrix
(H, W) = gray_img.shape
c = 0.5
d = get_filtering_matrix(H, W, c)

# Filter the noise image, multiplying it with the filtering matrix
hadamard_prod = np.multiply(noise_img, d)

# Return to the original domain
hf_noise = idct2(hadamard_prod)

# Plotting stuffs
f, (plot1, plot2, plot3) = plt.subplots(1, 3)

plot1.axis("off")
plot1.set_title("Original")
plot1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plot2.axis("off")
plot2.set_title("Noise image")
plot2.imshow(noise_img, cmap='gray')

plot3.axis("off")
plot3.set_title("HF Noise image")
plot3.imshow(hf_noise, cmap='gray')

plt.show()
