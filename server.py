from importlib import reload
import numpy as np
import os
import constants
import extract_dsnu
import enrollment

def enroll(enr_hf_noise):
    print("[INFO] Start enrollment")
    reload(enrollment)
    idx_bright, idx_dark = enrollment.get_indeces(enr_hf_noise)

    print(max( idx_bright ))
    print(min( idx_bright ))
    print(len( idx_bright ))
    print(max( idx_dark ))
    print(min( idx_dark ))
    print(len( idx_dark ))

    print("[INFO] Enrollment done")
    return idx_bright, idx_dark

def authenticate(idx_bright, idx_dark, auth_hf_noise):
    print("[INFO] Authentication")

    challenge = enrollment.get_challenge(idx_bright, idx_dark)
    print("[DEBUG] sending Challenge", challenge)
    ref_key = enrollment.get_reference_key(challenge, idx_bright)
    print("[DEBUG] reference key", ref_key)
    response = enrollment.get_response_key(auth_hf_noise.flatten(), challenge)
    print("[DEBUG] response key", response)

    if enrollment.are_equal(ref_key, response):
        print("Authentication completed")
    else:
        print("NEVER GONNA GIVE YOU UP")

test = False
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path_test = os.path.join(script_dir, 'downloads', 'raw', 'set-01', 'sensor-05', 'img-13.raw')
img_path_test_a = os.path.join(script_dir, 'downloads', 'raw', 'set-01', 'sensor-05', 'img-01.raw')

enroll_img = img_path_test
auth_img = img_path_test_a
# enroll_img = './images/IMG_5985.CR2'
# auth_img = './images/IMG_5986.CR2'

if not test:
    enr_hf_noise = extract_dsnu.get_hf_noise(enroll_img, False)
    auth_hf_noise = extract_dsnu.get_hf_noise(auth_img, False)

    print(enr_hf_noise == auth_hf_noise)
else:
    enr_hf_noise = np.array([9, 3, 0, 2, 5, 6, 14, 3, 1, 5, 2, 5, 11, 9, 6, 9, 8, 1, 10, 9, 8, 7, 10, 2])
    auth_hf_noise = np.array([9, 3, 0, 2, 5, 6, 14, 3, 1, 5, 2, 5, 11, 9, 6, 9, 8, 1, 10, 9, 8, 7, 10, 2])
    constants.num_blocks = 8
    constants.key_length = 8


while(True):
    i = input('> ')
    if i == 'e':
        idx_bright, idx_dark = enroll(enr_hf_noise)
    if i == 'a':
        authenticate(idx_bright, idx_dark, auth_hf_noise)
