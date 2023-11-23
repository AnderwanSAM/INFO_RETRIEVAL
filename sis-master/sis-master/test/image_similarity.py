#Importation des modules 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Fonction pour extraire les données de légende à partir d'un document
def extract_captions(document):
    extracted_data = []
    lines = document.split('\n') #On sépare ici en ligne
    for line in lines:
        parts = line.split(',') #On sépare ici chaque ligne en parties
        if len(parts) >= 2:
            image_name = parts[0].strip() #On extrait ici le nom de l'image et du texte après la virgule
            text_after_comma = parts[1].strip()
            extracted_data.append((image_name, text_after_comma))
    return extracted_data

#Fonction pour traiter une légende en supprimant les chiffres et en gardant le format minuscule
def process_caption(caption):
    caption = re.sub(r'\d+', '', caption) #Suppression chiffres
    return caption.lower() #Format minuscule

#Fonction pour créer des vecteurs TF-IDF à partir de légendes
def create_tfidf_vectors(captions):
    tfidf_vectorizer = TfidfVectorizer() #Initialisation du vectoriseur TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(captions) #On créer ici une matrice en se basant sur les légendes
    return tfidf_vectorizer, tfidf_matrix


#Fonction pour avoir les meilleures correspondances pour une requêtes donnée
def get_top_matches(query, tfidf_vectorizer, tfidf_matrix, captions, image_names):
    query_vector = tfidf_vectorizer.transform([query])

    #Similarité cosinus : mesure de similarité entre deux vecteurs non nuls d'un espace produit interne
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten() #Calcul des similarités entre la requête et les légendes
    caption_scores = list(enumerate(cosine_similarities))
    caption_scores.sort(key=lambda x: x[1], reverse=True) #Tri des légendes selon les scores de similarités (plus haut au plus bas)
    top_matches = caption_scores[:5] #On sélectionne ici les 5 meilleures correspondances

    #On créer ici une liste de résultats avec les noms d'image, les légendes et les scores
    result_matches = []
    for i, score in top_matches:
        result_matches.append((image_names[i], captions[i], score))

    return result_matches

#Fonction pour trouver et afficher les images similaires à la requête
def find_similar_images(query, captions_file_path):
    with open(captions_file_path, "r") as f: #On fait ici la lecture et l'extraction des données
        captions = extract_captions(f.read())

    processed_query = process_caption(query) #Traitement requête et création vecteurs
    tfidf_vectorizer, tfidf_matrix = create_tfidf_vectors([process_caption(c) for _, c in captions])
    
    #Meilleures correspondances
    top_matches = get_top_matches(processed_query, tfidf_vectorizer, tfidf_matrix, [c for _, c in captions], [i for i, _ in captions])

    #Affiche les résultats
    print(f"Top 5 similar images to the query '{query}':")
    for i, (image_name, caption, score) in enumerate(top_matches):
        print(f"Match {i + 1} - Score: {score:.4f}")
        print(f"Image Name: {image_name}, Caption: {caption}")
        print()