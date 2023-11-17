import cv2
import numpy as np
import os
import pandas as pd
from IPython.display import display
from PIL import Image

def get_vector(image, bins=32):
    red = cv2.calcHist(
        [image], [2], None, [bins], [0, 256]
    )
    green = cv2.calcHist(
        [image], [1], None, [bins], [0, 256]
    )
    blue = cv2.calcHist(
        [image], [0], None, [bins], [0, 256]
    )
    vector = np.concatenate([red, green, blue], axis=0)
    vector = vector.reshape(-1)
    return vector


def get_images_in_folder(folder_path):
    images_list = []
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a JPEG image
        if filename.endswith(".jpg"):
            # Create a full path to the image
            img_path = os.path.join(folder_path, filename)
            images_list.append(img_path)
    return images_list

def compute_vectors(folder_path, vector_computation_filename):
    list_of_images = get_images_in_folder(folder_path)
    vectors = []
    for filename in list_of_images: 
        image = cv2.imread(filename)
        if image is None:
            continue  # Skip if the image is not loaded properly
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
        vector = get_vector(image_rgb)
        vectors.append([filename, *vector])
    # Ensure vectors list is not empty
    if vectors:
        df = pd.DataFrame(vectors, columns=['image_id'] + [f'vec_{i}' for i in range(len(vectors[0])-1)])
        df.to_csv(vector_computation_filename, index=False)


def read_vectors_computation(vector_computation_filename): 
    df = pd.read_csv(vector_computation_filename)
    computed_vectors = df.set_index('image_id').to_dict('index')
    return computed_vectors

def find_image_vector(image_id,computed_vectors_dict): 
    return computed_vectors_dict.get(image_id)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


compute_vectors("static/img","colorhistogram_vectors.csv")
