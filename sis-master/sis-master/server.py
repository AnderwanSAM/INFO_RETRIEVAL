import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, render_template, request
from pathlib import Path
from main import get_top_matches, extract_captions, process_caption, create_tfidf_vectors

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
        
        # Save the query image
        uploaded_img_path = f"static/uploaded/{datetime.now().isoformat().replace(':', '_')}_{query_img.filename}"
        img.save(uploaded_img_path)

        # Run the image search
        query_features = fe.extract(img)
        dists = np.linalg.norm(features - query_features, axis=1)  
        ids = np.argsort(dists)[:30]  
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html', query_path=uploaded_img_path, scores=scores)
    else:
        return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/text', methods=['GET', 'POST'])
def text_search():
    if request.method == 'POST':
        query_text = request.form.get('query', '').strip()

        if not query_text:
            return render_template('text.html', error_message="Please enter a valid text query.")

        captions_file_path = "captions.txt"
        with open(captions_file_path, "r") as file:
            captions_and_image_names = extract_captions(file.read())

        processed_query = process_caption(query_text)
        tfidf_vectorizer, tfidf_matrix = create_tfidf_vectors([process_caption(caption) for _, caption in captions_and_image_names])

        image_names = [image_name for image_name, _ in captions_and_image_names]

        top_text_matches = get_top_matches(processed_query, tfidf_vectorizer, tfidf_matrix, [caption for _, caption in captions_and_image_names], image_names)

        return render_template('text.html', query_text=query_text, top_text_matches=top_text_matches)
    else:
        return render_template('text.html')

if __name__ == "__main__":
    app.run("0.0.0.0")
