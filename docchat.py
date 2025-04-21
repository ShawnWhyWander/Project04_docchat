import os
import re
import requests
from bs4 import BeautifulSoup
import PyPDF2

from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


api_key = os.getenv("GROQ_API_KEY")

def load_text(filepath_or_url):
    def is_url(path):
        return bool(urlparse(path).scheme in ('http','https'))

    if is_url(filepath_or_url):
        response = requests.get(filepath_or_url)
        if filepath_or_url.endswith('.txt'):
            return response.text
        elif filepath_or_url.endswith('.html'):
            soup = BeautifulSoup(response.text,'html.parser')
            return soup.get_text()
        elif filepath_or_url.endswith('.pdf'):
            with open('temp.pdf','wb') as f:
                f.write(response.content)
            with open('temp.pdf','rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
    else:
        if filepath_or_url.endswith(".txt"):
            with open(filepath_or_url, "r", encoding="utf-8") as f:
                return f.read()
        elif filepath_or_url.endswith(".html"):
            with open(filepath_or_url, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                return soup.get_text()
        elif filepath_or_url.endswith(".pdf"):
            with open(filepath_or_url, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])

    raise ValueError("Unsupported file format")


def chunk_text_by_words(text, max_words=100, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        chunks.append(" ".join(chunk))
        if i + max_words >= len(words):
            break
        i += max_words - overlap
    return chunks


def score_chunk(chunk, query):
    vectorizer = TfidfVectorize().fit([chunk, query])
    vectors = vectorizer.transform([chunk,query])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return score

def find_relevant_chunks(text, query, num_chunks=5):
    chunks = chunk_text_by_words(text)
    scored_chunks = [(chunk, score_chunk(chunk,query)) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:num_chunks]]
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()

