import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, render_template, request
from pathlib import Path
from main import get_top_matches, extract_captions, process_caption, create_tfidf_vectors
from colorhistogram_vectorize import read_vectors_computation, get_vector, cosine
from deep import st 
from text2 import get_top_matches2,train_bert_model,embed_captions

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []

# Define the paths for images and features
feature_path = Path("./static/feature")
img_path = Path("./static/img")

# Load image features
for feature_file in feature_path.glob("*.npy"):
    img_paths.append(f"static/img/{feature_file.stem}.jpg")
    features.append(np.load(feature_file))

features = np.array(features)

@app.route('/image', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle uploaded image
        query_img = request.files['query_img']
        img = Image.open(query_img.stream)
        
        # get the selected system 
        selected_system = request.form.get('search_system')

        # Save the query image
        uploaded_img_path = f"static/uploaded/{datetime.now().isoformat().replace(':', '_')}_{query_img.filename}"
        img.save(uploaded_img_path)

        # Run the image search

        if selected_system == "Color Histogram":
            # Process image using the colorH system
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            query_vector = get_vector(image)
            computed_vectors = read_vectors_computation('colorhistogram_vectors.csv')
            # Compute similarities
            scores = []
            for image_id, vectors in computed_vectors.items():
                vector = np.array([vectors[f'vec_{i}'] for i in range(len(query_vector))])
                similarity = cosine(query_vector, vector)
                scores.append((similarity, image_id))
            scores.sort(key=lambda x: x[0], reverse=True)  # Sort by similarity, highest first
            scores = scores[:10]  # Get top 10 matches
        elif selected_system == "VGG-16":
            # Process image using the features system
            query_features = fe.extract(img)
            dists = np.linalg.norm(features - query_features, axis=1)
            ids = np.argsort(dists)[:10]
            scores = [(dists[id], img_paths[id]) for id in ids]
            scores.sort(key=lambda x: x[0])
        elif selected_system == "Deep Image Search":
            similar_images = st.get_similar_images(image_path=uploaded_img_path, number_of_images=10)
            scores = list(similar_images.items())
            scores.sort(key=lambda x: x[0], reverse=True)

        return render_template('index.html', query_path=uploaded_img_path, scores=scores,selected_system=selected_system)
    else:
        return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/text', methods=['GET', 'POST'])
def text_search():
    matches = []
    if request.method == 'POST':
        query_text = request.form.get('query', '').strip()

        if not query_text:
            return render_template('text.html', error_message="Please enter a valid text query.")

        captions_file_path = "captions.txt"
        with open(captions_file_path, "r") as file:
            captions_and_image_names = extract_captions(file.read())

        processed_query = process_caption(query_text)
        selected_system2 = request.form.get('search_system2')
        
        if selected_system2 == "TF-IDF":
            print(processed_query)
            tfidf_vectorizer, tfidf_matrix = create_tfidf_vectors([process_caption(caption) for _, caption in captions_and_image_names])

            image_names = [image_name for image_name, _ in captions_and_image_names]

            matches = get_top_matches(processed_query, tfidf_vectorizer, tfidf_matrix, [caption for _, caption in captions_and_image_names], image_names)
            
        elif selected_system2 == "BERT model":
            bert_model = train_bert_model([c[1] for c in captions_and_image_names])
            
            caption_embeddings = embed_captions([process_caption(c[1]) for c in captions_and_image_names], bert_model)
            
            query_embedding = embed_captions([processed_query], bert_model)
            
            matches = get_top_matches2(query_embedding, caption_embeddings, [c[1] for c in captions_and_image_names], [c[0] for c in captions_and_image_names])


        return render_template('text.html', query_text=query_text, matches=matches,selected_system2=selected_system2)
            
    else:
        return render_template('text.html')

if __name__ == "__main__":
    app.run("0.0.0.0")
