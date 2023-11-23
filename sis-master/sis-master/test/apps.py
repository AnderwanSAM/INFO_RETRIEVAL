#Importation des modules 
from flask import Flask, render_template
from flask import request 

#On cree ici l'application Flask
app = Flask(__name__, template_folder='templates')

#Definition route de la page d'accueil
@app.route('/')
def home():
    return render_template('home.html') #Retour du template 'home.html'

#Definition route de la recherche d'images
@app.route('/image')
def image_search():
    return render_template('index.html') #Retour du template 'index.html'

#Definition route de la recherche de texte avec GET et POST
@app.route('/text', methods=['GET', 'POST'])
def text_search():
    if request.method == 'POST': #On vérifie ici la méthode de la requête
        query = request.form.get('query', '') # Si la méthode est POST, on aura la valeur du champ 'query' du formulaire
    return render_template('text.html') #Retour du template 'text.html'


