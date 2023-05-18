import numpy as np
import cv2

from extract_dsnu import get_hf_noise

img_path = './images/IMG_5985.CR2'

key_length = 128*2
compensation_margin = 0
num_blocks = key_length + compensation_margin

test = True
if not test:
    hf_noise = get_hf_noise(img_path, False)
    # Flatten the matrix
    flattened_hf_noise = hf_noise.flatten()
else:
    flattened_hf_noise = np.array([2, 5, 6, 14, 3, 1, 5, 2, 5, 11, 9, 6, 9, 8, 1, 10, 9, 8, 7, 10])
    num_blocks = 5

print('num_blocks:', num_blocks)
# Split the flattened matrix into blocks
blocks = np.array_split(flattened_hf_noise, num_blocks)

block_size = len(flattened_hf_noise) // num_blocks
print('block_size:', block_size)
# Get a list of tuples (maximum value, linear index, block index)
max_values = [(block.max(), block.argmax() + i * block_size, i) for i, block in enumerate(blocks)]

# Sort the list using the brightness as ordering key
sorted_values = sorted(max_values, key=lambda x: x[0])
# Select the brighter half
half_bright = sorted_values[-len(sorted_values) // 2:]
# Save the indices 
idx_brighter = [e[1] for e in half_bright]
print('idx_brighter', sorted(idx_brighter))

# Select the other half
less_brighter = sorted_values[:len(sorted_values) // 2]
# Search the darker pixels in those blocks
half_dark = [(blocks[i].min(), blocks[i].argmin() + i * block_size, i) for _,_,i in less_brighter]
# Select the linear indices of the darker pixels
idx_darker = [e[1] for e in half_dark]
print('idx_darker', sorted(idx_darker))
