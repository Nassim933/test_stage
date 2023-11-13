import re
import joblib
from PyPDF2 import PdfReader

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def modifie_text(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [sentence.capitalize() + '\n' for sentence in sentences]
    return ' '.join(sentences)

model = joblib.load('model.pkl')

def text_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def predict_text(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict([preprocessed_text])
    return prediction[0]

pdf_path = str(input("Veuillez saisir le nom du fichier pdf : "))


pdf_text = text_pdf(pdf_path)

texte_modifie = modifie_text(pdf_text)

pred = predict_text(texte_modifie)

if pred == 0:
    print("Le texte du PDF semble bien fait.")
else:
    print("Le texte du PDF semble mal fait.")

output_file_path = 'texte_modifie.txt'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(texte_modifie)

print(f"Le texte modifié a été enregistré dans : {output_file_path}")
