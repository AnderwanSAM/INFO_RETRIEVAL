# Ask for an image
# search image 
# print number image

import cv2
import numpy as np
import os

val = input("Enter the pathvalue: ")
print(val)

def get_and_validate_path_to_Collection():
    folderExists = False 
    path = ""
    while(not folderExists): 
        path = input("Enter the path to the collection: ")
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print("The folder exists. Moving on")
            folderExists = True
        else:
            print("The folder does not exist.")
    return path 

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


def get_queries():
    path_to_queries = "../../../../Benchmark/groundtruth/queries"
    list_of_images_queries = get_images_in_folder(path_to_queries)
    return list_of_images_queries

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


def create_vectors_list(collection_path):
    vectors_list = []
    list_of_images = get_images_in_folder(collection_path)
    for filename in list_of_images:
        # Read the image using OpenCV
        image = cv2.imread(filename)

        # Convert it to RGB (since OpenCV loads images in BGR format by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute the vector using your function
        vector = get_vector(image_rgb)

        # Append to the list
        vectors_list.append(vector)
    return vectors_list

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(idx,query_images_vectors,image_vectors, top_k=5):
    query_vector = query_images_vectors[idx]
    distances = []
    for _, vector in enumerate(image_vectors):
        distances.append(cosine(query_vector, vector))
    # get top k most similar images
    top_idx = np.argpartition(distances, -top_k)[-top_k:]
    return top_idx





    
    