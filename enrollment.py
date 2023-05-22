import random
import numpy as np

import constants

def get_indeces(img: np.ndarray):
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
    blocks = np.array_split(flattened_hf_noise, constants.num_blocks)

    block_size = len(flattened_hf_noise) // constants.num_blocks
    # Get a list of tuples (maximum value, linear index, block index)
    max_values = [(block.max(), block.argmax() + i * block_size, i) for i, block in enumerate(blocks)]

    # Sort the list using the brightness as ordering key
    sorted_values = sorted(max_values, key=lambda x: x[0]) # Select the brighter half
    half_bright = sorted_values[-len(sorted_values) // 2:]
    # Save the indices 
    idx_brighter = [e[1] for e in half_bright]

    # Select the other half
    less_brighter = sorted_values[:len(sorted_values) // 2]
    # Search the darker pixels in those blocks
    half_dark = [(blocks[i].min(), blocks[i].argmin() + i * block_size, i) for _,_,i in less_brighter]
    # Select the linear indices of the darker pixels
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
    print(distance)

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

    return hamming_distance(challenge, response) <= hamming_threshold

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

def get_response_key(hf_noise, challenge):
    m = [hf_noise[i] for i in challenge]
    median = np.median(m)
    response = [1 if i >= median else 0 for i in m]

    return response
