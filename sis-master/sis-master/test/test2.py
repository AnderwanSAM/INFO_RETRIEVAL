#Importation des modules
import unittest
from io import StringIO
from unittest.mock import patch
from image_similarity import extract_captions, process_caption, create_tfidf_vectors, get_top_matches, find_similar_images

#classe de tests pour les fonctions de image similarity
class TestImageSimilarity(unittest.TestCase):

    #Test de la fonction extract_captions
    def test_extract_captions(self):
        document = "image1, Caption 1\nimage2, Caption 2\nimage3, Caption 3" #On a ici des paires image-caption
        expected_output = [("image1", "Caption 1"), ("image2", "Caption 2"), ("image3", "Caption 3")] #Ce qu'on espère avoir après l'extraction
        self.assertEqual(extract_captions(document), expected_output) #On vérifie ici si le résultat attendu est produit

    #Test de la fonction process_caption
    def test_process_caption(self):
        caption = "This is a Test Caption 123" #Caption pour le test
        expected_output = "this is a test caption " #Caption attendu après le traitement
        self.assertEqual(process_caption(caption), expected_output) #On vérifie ici si le résultat attendu est produit

    #Test de la fonction create_tfidf_vectors
    def test_create_tfidf_vectors(self):
        captions = ["caption 1", "caption 2", "caption 3"] #Liste de caption
        tfidf_vectorizer, tfidf_matrix = create_tfidf_vectors(captions) #Création des vecteurs TF-IDF (Term Frequency-Inverse Document Frequency)

        #On vérifie ici que les vecteurs ne sont pas nuls
        self.assertIsNotNone(tfidf_vectorizer)
        self.assertIsNotNone(tfidf_matrix)

    #Test de la fonction get_top_matches
    def test_get_top_matches(self):
        query = "test query"

        #Creaction des vecteurs TF-IDF sur la base de captions de test
        tfidf_vectorizer, tfidf_matrix = create_tfidf_vectors(["test caption 1", "test caption 2", "test caption 3"])

        captions = ["test caption 1", "test caption 2", "test caption 3"]
        image_names = ["image1", "image2", "image3"]
        
        #Pour obtenir les meilleurs correspondances
        top_matches = get_top_matches(query, tfidf_vectorizer, tfidf_matrix, captions, image_names)
        expected_num_matches = min(5, len(captions)) #Calcul du nombre de correspondances attendues 
        self.assertEqual(len(top_matches), expected_num_matches) #On vérifie ici si les correspondances sont correct

    #Test de la fonction find_similar_images
    @patch('sys.stdout', new_callable=StringIO)
    def test_find_similar_images(self, mock_stdout):
        query = "test query"
        captions_file_path = "test_captions.txt"

        #On simule ici l'ouverture du fichier avec la fonction open
        with patch('builtins.open', return_value=StringIO("image1, test caption 1\nimage2, test caption 2\nimage3, test caption 3")):
            find_similar_images(query, captions_file_path)
        expected_output = "Top 5 similar images to the query 'test query':\n" #La sortie attentue
        self.assertIn(expected_output, mock_stdout.getvalue()) #Vérification de la sortie attendue

#Ce qui permet de run les tests
if __name__ == '__main__':
    unittest.main()
