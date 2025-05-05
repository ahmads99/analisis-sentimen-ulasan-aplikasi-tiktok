import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from collections import Counter

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load model
@st.cache_resource
def load_gru_model():
    return load_model('model/model_gru.h5')

model = load_gru_model()

# Load tokenizer (fit on the same data as during training)
@st.cache_resource
def load_tokenizer():
    vocab_limit = 2500
    tokenizer = Tokenizer(num_words=vocab_limit, split=' ')
    # Note: For simplicity, we assume the tokenizer was saved or retrained on the same dataset
    # Here, we fit on a placeholder (ideally, load a saved tokenizer)
    df = pd.read_csv('data/data_scraping_tiktok.csv')
    texts = df['content'].astype(str).tolist()
    tokenizer.fit_on_texts(texts)
    return tokenizer

tokenizer = load_tokenizer()

# Slang dictionary
slangwords = {
    'tdk': 'tidak', 'sy': 'saya', 'bgt': 'banget', 'nn': 'nanti', 'ga': 'tidak',
    'gak': 'tidak', 'tak': 'tidak', 'trus': 'terus', 'dikit': 'sedikit', 'ni': 'ini',
    'aja': 'saja', 'hp': 'handphone', 'gmn': 'gimana', 'bgmn': 'bagaimana', 'kmn': 'kemana',
    'sih': 'si', 'sm': 'sama', 'bs': 'bisa', 'klo': 'kalau', 'kl': 'kalau', 'dr': 'dari',
    'dg': 'dengan', 'tp': 'tapi', 'blm': 'belum', 'udh': 'sudah', 'ud': 'sudah', 'lg': 'lagi',
    'skrg': 'sekarang', 'brp': 'berapa', 'eror': 'error', 'erorr': 'error', 'err': 'error',
    'bgus': 'bagus', 'bgs': 'bagus', 'lemot': 'lambat', 'lambt': 'lambat', 'aplk': 'aplikasi',
    'apk': 'aplikasi', 'app': 'aplikasi', 'kzl': 'kesal', 'jgn': 'jangan', 'bkn': 'bukan',
    'gt': 'gitu', 'cb': 'coba', 'parah': 'sangat', 'jos': 'bagus', 'top': 'bagus',
    'mantab': 'mantap', 'ok': 'oke', 'oky': 'oke', 'sip': 'oke', 'tlg': 'tolong',
    'thx': 'terima kasih', 'tx': 'terima kasih', 'pls': 'tolong', 'plis': 'tolong',
    'bgtz': 'banget', 'bener': 'benar', 'bner': 'benar', 'tb': 'tiba', 'tba': 'tiba',
    'gk': 'tidak', 'nggak': 'tidak', 'ngk': 'tidak', 'gajelas': 'tidak jelas',
    'ngebuk': 'bug', 'ngestuk': 'stuck', 'nyetuck': 'stuck', 'ngadat': 'macet',
    'tt': 'tiktok', 'anj': 'anjing', 'euy': 'euy', 'smpa': 'sumpah', 'udah': 'sudah',
    'gbs': 'tidak bisa', 'gabisa': 'tidak bisa', 'ngga': 'tidak', 'kpn': 'kapan',
    'msk': 'masuk', 'cm': 'cuma', 'bbrp': 'beberapa', 'lgsg': 'langsung', 'smp': 'sampai',
    'mskpn': 'meskipun', 'krn': 'karena', 'kdg': 'kadang', 'kdng': 'kadang', 'skli': 'sekali',
    'smpe': 'sampai', 'ny': 'nya', 'sbnrnya': 'sebenarnya', 'slalu': 'selalu', 'smua': 'semua',
    'stlh': 'setelah', 'kcl': 'kecil', 'bnyk': 'banyak', 'trsprh': 'terus terang', 'mskpn': 'meskipun',
    'benerin': 'perbaiki', 'pls': 'tolong', 'pliss': 'tolong', 'sblm': 'sebelum',
    'kli': 'kali', 'bbrapa': 'beberapa', 'lg': 'lagi', 'msk': 'masuk', 'km': 'kamu'
}

# Text preprocessing functions
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    words = text.split()
    fixed_words = [slangwords.get(word.lower(), word) for word in words]
    return ' '.join(fixed_words)

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(text):
    list_stopwords = set(stopwords.words('indonesian')).union(set(stopwords.words('english')))
    additional_stopwords = [
        'ini', 'saya', 'nya', 'si', 'aja', 'ajah', 'cuma', 'kok', 'tuh', 'doang',
        'dah', 'eh', 'ko', 'lah', 'deh', 'ya', 'nih', 'gitu', 'kmn', 'tpi',
        'gue', 'aku', 'ku', 'dong', 'dlu', 'lgi', 'banget', 'sama', 'buat', 'kalo',
        'mau', 'pas', 'udah', 'terus', 'tadi', 'skrng',
        'admin', 'adminnya', 'min', 'minnya', 'sobat', 'kakak', 'bro', 'sis',
        'guys', 'gaes', 'gaess', 'om', 'tante',
        'woy', 'woi', 'hai', 'cuy', 'lo', 'loh', 'plis', 'woe', 'hallo', 'euy',
        'anjir', 'njir', 'sih', 'tau', 'lho', 'btw', 'cmn', 'yaa', 'yaaa', 'yh',
        'gmn', 'gmana', 'gmna'
    ]
    list_stopwords.update(additional_stopwords)
    return [word for word in text if word not in list_stopwords]

def toSentence(list_words):
    return ' '.join(list_words)

# Preprocess input text
def preprocess_text(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = fix_slangwords(text)
    tokens = tokenizingText(text)
    tokens = filteringText(tokens)
    text = toSentence(tokens)
    return text

# Ini harus menjadi perintah Streamlit pertama di skrip Anda
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

# Modern UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .title {
        font-size: 2.5em;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .result {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        border-radius: 5px;
    }
    .positive { background-color: #d4edda; color: #155724; }
    .negative { background-color: #f8d7da; color: #721c24; }
    .neutral { background-color: #e2e3e5; color: #383d41; }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title">TikTok Review Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan ulasan untuk memprediksi sentimennya (Positive, Negative, or Neutral)</div>', unsafe_allow_html=True)

# Text input
user_input = st.text_area("Masukkan ulasan Anda:", height=150, placeholder="Type your TikTok review here...")

# Predict button
if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess input
        processed_text = preprocess_text(user_input)
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=100)  # Adjust maxlen to match training

        # Predict
        prediction = model.predict(padded)
        categorical_class = ["negative", "positive", "neutral"]
        predicted_label = categorical_class[np.argmax(prediction, axis=1)[0]]

        # Display result with styling
        if predicted_label == "positive":
            st.markdown(f'<div class="result positive">Sentiment: Positive</div>', unsafe_allow_html=True)
        elif predicted_label == "negative":
            st.markdown(f'<div class="result negative">Sentiment: Negative</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result neutral">Sentiment: Neutral</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a review text.")

# Footer
st.markdown("<hr><div style='text-align: center; color: #666;'>Powered by Streamlit & TensorFlow</div>", unsafe_allow_html=True)