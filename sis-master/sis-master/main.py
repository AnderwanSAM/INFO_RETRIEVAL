import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_captions(document):
    extracted_data = []
    lines = document.split('\n')
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 2:
            image_name = parts[0].strip()
            text_after_comma = parts[1].strip()
            extracted_data.append((image_name, text_after_comma))
    return extracted_data

def process_caption(caption):
    caption = re.sub(r'\d+', '', caption) 
    return caption.lower()

def create_tfidf_vectors(captions):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(captions)
    return tfidf_vectorizer, tfidf_matrix

def get_top_matches(query, tfidf_vectorizer, tfidf_matrix, captions, image_names):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    caption_scores = list(enumerate(cosine_similarities))
    caption_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = caption_scores[:5]
    
    result_matches = []
    for i, score in top_matches:
        result_matches.append((image_names[i], captions[i], score))
    
    return result_matches

def find_similar_images(query, captions_file_path):
    with open(captions_file_path, "r") as f:
        captions = extract_captions(f.read())

    processed_query = process_caption(query)
    print(processed_query)
    tfidf_vectorizer, tfidf_matrix = create_tfidf_vectors([process_caption(c) for c in captions])
    top_matches = get_top_matches(processed_query, tfidf_vectorizer, tfidf_matrix, captions)

    print(f"Top 5 similar images to the query '{query}':")
    for i, (caption, score) in enumerate(top_matches):
        print(f"Match {i + 1} - Score: {score:.4f}")
        print(f"Caption: {caption}")
        print()
