import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import pdfminer.high_level
from pdfminer.layout import LAParams
import os
import spacy
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK resources (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def extract_text_from_pdf(pdf_path):
    try:
        output_string = StringIO()
        with open(pdf_path, 'rb') as in_file:
            pdfminer.high_level.extract_text_to_fp(in_file, output_string, laparams=LAParams())
        return output_string.getvalue()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return " ".join(filtered_tokens)


def extract_name(resume_text):
    doc = nlp(resume_text)
    for ent in
