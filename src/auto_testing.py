import logging
import os
import constants
import extract_dsnu
import server

# PATH VARIABLES
img_test_enroll = os.path.join(constants.dataset_path, "set-01", "sensor-01", "img-01.raw")
img_test_auth_dir = os.path.join(constants.dataset_path, "set-02", "sensor-01")

try:
    # Enrollment
    logging.info("[TESTING] beginning enrollment...")
    logging.info(f"[TESTING] current Hamming Distance threshold: {constants.hamming_threshold}")
    enroll_img = img_test_enroll
    enr_hf_noise = extract_dsnu.get_hf_noise(enroll_img, constants.dataset_width, constants.dataset_height)
    idx_bright, idx_dark = server.enroll(enr_hf_noise)
    logging.info("[TESTING] enrollment done")

    # Auth testing
    logging.info("[TESTING] beginning auth testing...")

    file_list = os.listdir(img_test_auth_dir)
    file_count = len(file_list)
    i = 0
    auth_count = 0
    hd_tot = 0

    for filename in file_list:
        i += 1
        auth_img = os.path.join(img_test_auth_dir, filename)
        auth_hf_noise = extract_dsnu.get_hf_noise(auth_img, constants.dataset_width, constants.dataset_height)

        auth, hd = server.authenticate(idx_bright, idx_dark, auth_hf_noise)
        hd_tot += hd
        if auth:
            auth_count += 1
            logging.info(f"[TESTING] {filename} ({i}/{file_count}): AUTHENTICATED (HD: {hd})")
        else:
            logging.info(f"[TESTING] {filename} ({i}/{file_count}): NOT AUTHENTICATED (HD: {hd})")

    # Results
    hd_avg = hd_tot / i
    logging.info("[TESTING] auth testing done")
    logging.info(f"[TESTING] authenticated images: {auth_count}/{i}")
    logging.info(f"[TESTING] average Hamming Distance: {hd_avg}")

except Exception as e:
    logging.info(f"[ERROR] an error occurred: {e}")