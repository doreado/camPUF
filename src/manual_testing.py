import logging
import server
import constants
import extract_dsnu
import os
import numpy as np

test = False

# Images
enroll_img = [
    os.path.join(constants.dataset_path, "set-01", "sensor-01", "img-01.raw"),
    os.path.join(constants.dataset_path, "set-01", "sensor-01", "img-02.raw"),
    os.path.join(constants.dataset_path, "set-01", "sensor-01", "img-03.raw"),
    os.path.join(constants.dataset_path, "set-01", "sensor-01", "img-04.raw"),
    os.path.join(constants.dataset_path, "set-01", "sensor-01", "img-05.raw"),
]
auth_img = [
    os.path.join(constants.dataset_path, "set-02", "sensor-01", "img-11.raw"),
    os.path.join(constants.dataset_path, "set-02", "sensor-01", "img-12.raw"),
]

if not test:
    enr_hf_noise = extract_dsnu.get_hf_noise(enroll_img, constants.dataset_width, constants.dataset_height)
    auth_hf_noise = extract_dsnu.get_hf_noise(auth_img, constants.dataset_width, constants.dataset_height)

    print(enr_hf_noise == auth_hf_noise)
else:
    enr_hf_noise = np.array([9, 3, 0, 2, 5, 6, 14, 3, 1, 5, 2, 5, 11, 9, 6, 9, 8, 1, 10, 9, 8, 7, 10, 2])
    auth_hf_noise = np.array([9, 3, 0, 2, 5, 6, 14, 3, 1, 5, 2, 5, 11, 9, 6, 9, 8, 1, 10, 9, 8, 7, 10, 2])
    constants.num_blocks = 8
    constants.key_length = 8

while(True):
    i = input('> ')
    if i == 'e':
        idx_bright, idx_dark = server.enroll(enr_hf_noise)
    if i == 'a':
        a, hd = server.authenticate(idx_bright, idx_dark, auth_hf_noise)
        logging.info(f"[INFO] Hamming Distance: {hd}")