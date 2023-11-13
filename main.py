import re
import joblib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

bon_texte = [read_text_from_file('bien.txt')]
mauvais_texte = [read_text_from_file('mal.txt')]

bon_textes = [preprocess_text(text) for text in bon_texte]
mauvais_textes = [preprocess_text(text) for text in mauvais_texte]

labels_bien_faits = [0] * len(bon_textes)
labels_mal_faits = [1] * len(mauvais_textes)

X_train = bon_textes + mauvais_textes
y_train = labels_bien_faits + labels_mal_faits

model = make_pipeline(TfidfVectorizer(), SVC())

model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')


