import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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

def train_bert_model(captions):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

def embed_captions(captions, bert_model):
    return bert_model.encode(captions, convert_to_tensor=True)

def get_top_matches2(query_embedding, caption_embeddings, captions, image_names):
    cosine_similarities = cosine_similarity(query_embedding, caption_embeddings).flatten()
    caption_scores = list(enumerate(cosine_similarities))
    caption_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = caption_scores[:10]

    result_matches = []
    for i, score in top_matches:
        result_matches.append((image_names[i], captions[i], score))

    return result_matches

def find_similar_images(query, captions_file_path):
    with open(captions_file_path, "r") as f:
        captions = extract_captions(f.read())

    processed_query = process_caption(query)

    # Train BERT model
    bert_model = train_bert_model([c[1] for c in captions])

    # Embed captions using BERT
    caption_embeddings = embed_captions([process_caption(c[1]) for c in captions], bert_model)
    query_embedding = embed_captions([processed_query], bert_model)

    tfidf_vectorizer, tfidf_matrix = create_tfidf_vectors([process_caption(c[1]) for c in captions])
    top_matches = get_top_matches2(query_embedding, caption_embeddings, [c[1] for c in captions], [c[0] for c in captions])

    print(f"Top 10 similar images to the query '{processed_query}':")
    for i, (image_name, caption, score) in enumerate(top_matches):
        print(f"Match {i + 1} - Score: {score:.4f}")
        print(f"Image Name: {image_name}")
        print(f"Caption: {caption}")
        print()


