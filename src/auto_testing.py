import logging
import os
import constants
import extract_dsnu
import server

# PATH VARIABLES
img_test_enroll_dir = os.path.join(constants.dataset_path, "set-02", "sensor-01")
width_enroll = constants.dataset_width
height_enroll = constants.dataset_height
num_frames_enroll = 5

img_test_auth_dir = os.path.join(constants.dataset_path, "set-01", "sensor-01")
width_auth = constants.dataset_width
height_auth = constants.dataset_height

try:
    # Enrollment
    file_list_enr = os.listdir(img_test_enroll_dir)
    file_count_enr = len(file_list_enr)
    if file_count_enr < num_frames_enroll:
        raise ValueError("[ERROR] more samples needed!")
    
    logging.info(f"[TESTING] beginning enrollment on {num_frames_enroll} frames...")
    logging.info(f"[TESTING] current Hamming Distance threshold: {constants.hamming_threshold}")

    # Multiple samples
    enroll_images = file_list_enr[:num_frames_enroll]
    for i in range(len(enroll_images)):
        enroll_images[i] = os.path.join(img_test_enroll_dir, enroll_images[i])
        
    enr_hf_noise = extract_dsnu.get_hf_noise(enroll_images, width_enroll, height_enroll)
    idx_bright, idx_dark = server.enroll(enr_hf_noise)
    logging.info("[TESTING] enrollment done")

    # Auth testing
    logging.info("[TESTING] beginning auth testing...")

    file_list_auth = os.listdir(img_test_auth_dir)
    file_count_auth = len(file_list_auth)
    i = 0
    auth_count = 0
    hd_tot = 0

    for filename in file_list_auth:
        i += 1
        auth_img = os.path.join(img_test_auth_dir, filename)
        auth_hf_noise = extract_dsnu.get_hf_noise([auth_img], width_auth, height_auth)

        auth, hd = server.authenticate(idx_bright, idx_dark, auth_hf_noise)
        hd_tot += hd
        if auth:
            auth_count += 1
            logging.info(f"[TESTING] {filename} ({i}/{file_count_auth}): AUTHENTICATED (HD: {hd})")
        else:
            logging.info(f"[TESTING] {filename} ({i}/{file_count_auth}): NOT AUTHENTICATED (HD: {hd})")

    # Results
    hd_avg = hd_tot / i
    logging.info("[TESTING] auth testing done")
    logging.info(f"[TESTING] authenticated images: {auth_count}/{i}")
    logging.info(f"[TESTING] average Hamming Distance: {hd_avg}")

except Exception as e:
    logging.info(f"[ERROR] an error occurred: {e}")