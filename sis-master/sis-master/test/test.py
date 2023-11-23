#Importtion des modules
import unittest
from flask_testing import TestCase
from flask import Flask
from flask.testing import FlaskClient
from apps import app

#Classe de tests qui hérite de TestCase
class TestApp(TestCase):

    #méthode permettant de créer l'application Flask pour réaliser les tests
    def create_app(self): 
        app.config['TESTING'] = True #mode de test pour configurer l'application
        return app

    #Test de la page d'accueil
    def test_home_page(self):
        response = self.client.get('/') #Envoi de la requête GET
        self.assert200(response) #On vérifie que le code HTTP est 200(OK)
        self.assertTemplateUsed('home.html') #On vérifie que le template 'home.html' est utilisé

    #Test de la page de recherche d'images
    def test_image_search_page(self):
        response = self.client.get('/image')
        self.assert200(response)
        self.assertTemplateUsed('index.html')

    #Test de la page de recherche de texte
    def test_text_search_page(self):
        response = self.client.get('/text')
        self.assert200(response)
        self.assertTemplateUsed('text.html')

    #Test de la page de recherche de texte avec une requête valide
    def test_text_search_with_valid_query(self):
        response = self.client.post('/text', data=dict(query='valid_query'))
        self.assert200(response)
        self.assertTemplateUsed('text.html')

    #Test de la page de recherche de texte avec une requête vide
    def test_text_search_with_empty_query(self):
        response = self.client.post('/text', data=dict(query=''))
        self.assert200(response)
        self.assertTemplateUsed('text.html')

#Ce qui permet de run les tests
if __name__ == '__main__':
    unittest.main()
