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
    height, width = img.shape
    print("[DEBUG] the noise image has", height, "x", width)
    print("[DEBUG] having an noise image of", len(flattened_hf_noise), "pixels")
    print("[DEBUG] dividing the noise image in", constants.num_blocks, "blocks")
    # Split the flattened matrix into blocks
    # if the length of the image is not a multiple of num_blocks
    # an element is added to the first length % num_blocks blocks
    blocks = np.array_split(flattened_hf_noise, constants.num_blocks)

    block_size = np.uint64(np.ceil(len(flattened_hf_noise) / constants.num_blocks))
    print("[DEBUG] working with a block size of", block_size, "pixels")
    print("[DEBUG] the first", len(flattened_hf_noise) % constants.num_blocks, "are", block_size + 1, "long")
    # Get a list of tuples (maximum value, linear index, block index)

    max_values = []
    for i, block in enumerate(blocks):
        count = 0
        print("[", i, "]")
        if i < (len(flattened_hf_noise) % constants.num_blocks):
            actual_block_size = (block_size+1)
        else:
            actual_block_size = block_size

        try:
            assert block.max() == flattened_hf_noise[np.uint64(block.argmax() + i * actual_block_size)]
        except:
            index = np.uint64(block.argmax() + i * actual_block_size)
            print("got index", index, "for", block.max())

            print("10 element forward")
            for j in range(1, 10):
                if block.max() == flattened_hf_noise[np.uint64(block.argmax() + (i * actual_block_size) + j)]:
                    print(".", j, ".", "hit at" , np.uint64(block.argmax() + (i * actual_block_size) + j))

            print("10 element backward")
            for j in range(1, 10):
                if block.max() == flattened_hf_noise[np.uint64(block.argmax() + (i * actual_block_size) - j)]:
                    print(".", j, ".", "hit at" , np.uint64(block.argmax() + (i * actual_block_size) - j))

        max_values.append((
            np.uint8(block.max()),
            np.uint64(block.argmax() + i * actual_block_size),
            i
        ))

    # print("max_values")
    # print(max_values)
    # Sort the list using the brightness as ordering key
    sorted_values = sorted(max_values, key=lambda x: x[0]) # Select the brighter half
    # TODO use np.split to split the array in two
    # half_bright, half_dark = np.split(sorted_values, 2)

    # half_bright = sorted_values[-len(sorted_values) // 2 - 1:]
    half_bright = sorted_values[len(sorted_values) // 2 : len(sorted_values)]
    # print("HALF BRIGHTER")
    # print(half_bright)
    # Save the indices 
    idx_brighter = [e[1] for e in half_bright]

    # Select the other half
    less_brighter = sorted_values[:len(sorted_values) // 2]
    # Search the darker pixels in those blocks
    half_dark = []
    for _,_,i in less_brighter:
        if i < (len(flattened_hf_noise) % constants.num_blocks):
            actual_block_size = (block_size+1)
        else:
            actual_block_size = block_size

        assert blocks[i].min() == flattened_hf_noise[np.uint64(blocks[i].argmax() + i * actual_block_size)]

        half_dark.append((
            np.uint8(blocks[i].min()),
            np.uint64(blocks[i].argmin() + i * actual_block_size),
            i
        ))

    # print("HALF DARK")
    # print(half_dark)
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
    print("[DEBUG] distance", distance)

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
    print(m)
    print(median)
    response = [1 if i >= median else 0 for i in m]

    return response
