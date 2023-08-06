import random
import numpy as np
import constants
import logging

def enroll(enr_hf_noise):
    logging.debug("[DEBUG] Start enrollment")
    idx_bright, idx_dark = get_indices(enr_hf_noise)

    logging.debug(max( idx_bright ))
    logging.debug(min( idx_bright ))
    logging.debug(len( idx_bright ))
    logging.debug(max( idx_dark ))
    logging.debug(min( idx_dark ))
    logging.debug(len( idx_dark ))

    logging.debug("[DEBUG] Enrollment done")
    return idx_bright, idx_dark

def authenticate(ref_key, response):
    logging.debug(f"[DEBUG] reference key {ref_key}")

    eq, hd = are_equal(ref_key, response)
    if eq:
        logging.debug("[DEBUG] Authentication completed")
        return True, hd
    else:
        logging.debug("[DEBUG] Authentication failed")
        return False, hd

def get_linear_index(block_index, block_size, offset, rem):
    # the first rem elements have an additional element
    # so the next ones must consider it
    if block_index > rem: # (hf_length % constants.num_blocks):
        offset = offset + rem

    return block_index * block_size + offset

def get_indices(img: np.ndarray):
    """
    Get the indices of brighter and darker pixels in an image.

    Args:
        img (np.ndarray): Input image as a NumPy array.

    Returns:
        Tuple[List[int], List[int]]: Tuple containing two lists:
        - idx_brighter: Indices of brighter pixels in the image.
        - idx_darker: Indices of darker pixels in the image.
    """

    flattened_hf_noise = img.flatten()
    # Split the flattened matrix into blocks
    # if the length of the image is not a multiple of num_blocks
    # an element is added to the first length % num_blocks blocks
    blocks = np.array_split(flattened_hf_noise, constants.num_blocks)
    rem = len(flattened_hf_noise) % constants.num_blocks

    # Get a list of tuples (maximum value, linear index, block index)
    max_values = [(
        np.uint8(block.max()),
        np.uint64(get_linear_index(i, len(block), block.argmax(), rem)),
        i
    ) for i, block in enumerate(blocks)]

    sorted_values = sorted(max_values, key=lambda x: x[0])
    # TODO use np.split to split the array in two
    # half_bright, half_dark = np.split(sorted_values, 2)

    half_bright = sorted_values[len(sorted_values) // 2 : len(sorted_values)]
    idx_brighter = [e[1] for e in half_bright]

    # Select the other half
    less_brighter = sorted_values[:len(sorted_values) // 2]
    # Search the darker pixels in those blocks
    half_dark = [(
            np.uint8(blocks[i].min()),
            np.uint64(get_linear_index(i, len(blocks[i]), blocks[i].argmin(), rem)),
            i
    ) for _,_,i in less_brighter ]

    idx_darker = [e[1] for e in half_dark]

    return idx_brighter, idx_darker

def hamming_distance(a, b):
    """
    Calculate the Hamming distance between two lists.

    Args:
        challenge (List[int]): List containing the challenge values.
        response (List[int]): List containing the response values.
    Returns:
        int: The Hamming distance between the challenge and response lists.

    Raises:
        ValueError: If the input lists have different lengths.
    """

    if len(a) != len(b):
        raise ValueError("Input lists must have the same length.")

    distance = sum(bit1 != bit2 for bit1, bit2 in zip(a, b))
    return distance

def are_equal(challenge: list[int], response: list[int], hamming_threshold = constants.hamming_threshold):
    """
    Check if the Hamming distance between the challenge and response is below or equal to the threshold.

    Args:
        challenge (List[int]): List containing the challenge values.
        response (List[int]): List containing the response values.
        hamming_threshold (int, optional): Hamming distance threshold. Defaults to constants.hamming_threshold.

    Returns:
        bool: True if the Hamming distance is below or equal to the threshold, False otherwise.
    """
    return hamming_distance(challenge, response) <= hamming_threshold, hamming_distance(challenge, response)

def get_reference_key(challenge, idx_bright):
    # Get the reference key
    ref_key = [1 if np.isin(idx_bright, i).any() else 0 for i in challenge]

    return ref_key

def get_challenge(idx_bright, idx_dark):
    if (len(idx_bright) != len(idx_dark)):
        raise ValueError("indeces list must have the same length")

    # Select L/2 entries from idx_dark
    random_idx_dark = np.array(random.sample(idx_dark, constants.key_length // 2))
    # Select L/2 entries from idx_bright
    random_idx_bright = np.array(random.sample(idx_bright, constants.key_length // 2))
    # Sort the sequence by index
    challenge = np.sort(np.concatenate((random_idx_dark, random_idx_bright)))

    return challenge
