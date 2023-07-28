from importlib import reload
import enrollment
import logging

def enroll(enr_hf_noise):
    logging.debug("[DEBUG] Start enrollment")
    reload(enrollment)
    idx_bright, idx_dark = enrollment.get_indeces(enr_hf_noise)

    logging.debug(max( idx_bright ))
    logging.debug(min( idx_bright ))
    logging.debug(len( idx_bright ))
    logging.debug(max( idx_dark ))
    logging.debug(min( idx_dark ))
    logging.debug(len( idx_dark ))

    logging.debug("[DEBUG] Enrollment done")
    return idx_bright, idx_dark

def authenticate(idx_bright, idx_dark, auth_hf_noise):
    logging.debug("[DEBUG] Authentication")

    challenge = enrollment.get_challenge(idx_bright, idx_dark)
    logging.debug(f"[DEBUG] sending Challenge {challenge}")
    ref_key = enrollment.get_reference_key(challenge, idx_bright)
    logging.debug(f"[DEBUG] reference key {ref_key}")
    response = enrollment.get_response_key(auth_hf_noise.flatten(), challenge)
    logging.debug(f"[DEBUG] response key {response}")

    eq, hd = enrollment.are_equal(ref_key, response)
    if eq:
        logging.debug("[DEBUG] Authentication completed")
        return True, hd
    else:
        logging.debug("[DEBUG] Authentication failed")
        return False, hd