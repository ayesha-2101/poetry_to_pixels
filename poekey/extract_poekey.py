from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def extract_emotion(summary):
    print(emotion_classifier(summary))
    return emotion_classifier(summary)[0][0]['label']


nlp = spacy.load("en_core_web_sm")

def extract_visuals(summary):
    doc = nlp(summary)
    return [chunk.text for chunk in doc.noun_chunks]


model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data/themes.txt") as f:
    themes = [line.strip() for line in f.readlines()]

theme_embeddings = model.encode(themes)

def extract_theme(summary):
    summary_emb = model.encode(summary)
    sims = util.cos_sim(summary_emb, theme_embeddings)[0]
    return themes[np.argmax(sims)]