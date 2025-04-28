import os
import langid
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import io
import soundfile as sf
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
from PIL import Image
import pytesseract


import json
import requests
from bs4 import BeautifulSoup
import PyPDF2

from urllib.parse import urlparse



load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


def load_text(filepath_or_url):
    '''
    For local files:
    >>> with open("sample.txt", "w") as f:
    ...     _ = f.write("Hello world!")
    >>> load_text("sample.txt")
    'Hello world!'

    >>> with open("sample.html", "w") as f:
    ...     _ = f.write("<html><body><h1>Test</h1><p>More</p></body></html>")
    >>> load_text("sample.html")
    'TestMore'

    >>> text = load_text("https://documents1.worldbank.org/curated/en/964051593660119868/pdf/Responding-to-COVID-19-in-China-A-Case-Study-of-Shanghai.pdf")
    >>> isinstance(text, str)
    True

    >>> load_text("unsupported_file.xyz")
    Traceback (most recent call last):
    ...
    ValueError: Unsupported file format
    '''
    if os.path.exists(filepath_or_url):
        if filepath_or_url.lower().endswith((".png", ".jpg", ".jpeg")):
            print("🖼️ Loading image and extracting text...")
            image = Image.open(filepath_or_url)
            text = pytesseract.image_to_string(image)
            return text
        elif filepath_or_url.lower().endswith(".html"):
            with open(filepath_or_url, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, 'html.parser')
                return soup.get_text()
        elif filepath_or_url.lower().endswith(".pdf"):
            with open(filepath_or_url, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
        elif filepath_or_url.lower().endswith(".txt"):
            with open(filepath_or_url, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {filepath_or_url}")
    else:
        if not (filepath_or_url.startswith("http://") or filepath_or_url.startswith("https://")):
            raise ValueError("Unsupported file format")
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(filepath_or_url, headers=headers)

        content_type = response.headers.get('Content-Type', '').lower()

        if 'text/html' in content_type:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
        elif 'application/pdf' in content_type:
            with open('temp.pdf', 'wb') as f:
                f.write(response.content)
            with open('temp.pdf', 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
        elif 'text/plain' in content_type:
            return response.text
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

def summarize_document(document_text):
    '''
    Summarizes the document.

    Example (runs without errors):
    >>> # summarize_document("This is a long document about economics.")
    '''


    prompt = f"""
    Summarize the following document briefly (in 3-5 sentences):

    {document_text[:4000]}  # Only send first 4000 characters to avoid overloading
    """

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 300
        }
    )

    try:
        summary = response.json()["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        print("❌ Failed to summarize document:", e)
        print(response.text)
        return "No summary available."


def chunk_text_by_words(text, max_words=100, overlap=50):
    '''
    >>> chunk_text_by_words("one two three four five six", max_words=3, overlap=1)
    ['one two three', 'three four five', 'five six']

    >>> chunk_text_by_words("hello world", max_words=5, overlap=2)
    ['hello world']

    >>> chunk_text_by_words("", max_words=4, overlap=2)
    []

    >>> chunk_text_by_words("a b c d e f g h", max_words=4, overlap=2)
    ['a b c d', 'c d e f', 'e f g h']
    '''
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
    '''
    >>> 0.45 <= score_chunk("the cat sat on the mat", "cat on mat") <= 0.50
    True

    >>> 0.0 <= score_chunk("the dog barked loudly", "cat on mat") <= 0.01
    True

    >>> 0.0 <= score_chunk("apple banana grape", "yellow fruit") <= 0.01
    True

    >>> 0.25 <= score_chunk("bananas are yellow", "yellow fruit") <= 0.30
    True
    '''
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([chunk, query])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0].item()

def expand_query_with_synonyms(query):
    '''
    Expands a query with synonyms.

    Example (runs without errors):
    >>> # expand_query_with_synonyms("cat")
    '''
    prompt = f"""
    List 5 short synonyms or related words for the following query:
    {query}
    Only list the words separated by spaces. No explanations.
    """
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 50
        }
    )
    try:
        synonyms = response.json()["choices"][0]["message"]["content"].strip()
        return f"{query} {synonyms}"
    except:
        print("❌ Synonym expansion failed")
        return query


def find_relevant_chunks(text, query, num_chunks=5):
    """
    Examples:
    >>> find_relevant_chunks("The cat sat on the mat. The dog barked loudly.", "cat", num_chunks=1)
    ['The cat sat on the mat. The dog barked loudly.']

    >>> find_relevant_chunks("Apples are red. Bananas are yellow. Grapes are purple.", "yellow fruit", num_chunks=1)
    ['Apples are red. Bananas are yellow. Grapes are purple.']

    >>> find_relevant_chunks("", "anything", num_chunks=1)
    []

    >>> find_relevant_chunks("sun moon earth", "stars", num_chunks=2)
    ['sun moon earth']
    """
    expanded_query = expand_query_with_synonyms(query)
    chunks = chunk_text_by_words(text)
    scored_chunks = [(chunk, score_chunk(chunk,expanded_query)) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:num_chunks]]



def maybe_translate_query(query,document_text):
    '''
    Maybe translates the query if document is not in English.

    Example (runs without errors):
    >>> # maybe_translate_query("What is this?", "Hello world")
    '''
    doc_lang, confidence = langid.classify(document_text)

    if doc_lang != "en" and confidence > 0.85:
        print(f"Detected document language: {doc_lang} (confidence: {confidence:.2f})")
        query = translate_query(query, doc_lang)
    else:
        print(f"Treating document as English (lang: {doc_lang}, confidence: {confidence:.2f})")

    translation_prompt = f"""
    Translate the following English question into {doc_lang}:
    {query}
    Only return the translated question.
    """
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer " + api_key,
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": translation_prompt}],
            "temperature": 0.0,
            "max_tokens": 100            
        }
    )

    try:
        return response.json()["Choices"][0]["message"]["content"].strip()
    except:
        print ("Translation failed")
        return query



def speak_text(text):
    """
    Converts the given text into speech using Groq's TTS API
    and plays the audio out loud.

    This function plays audio and does not return a value,
    so it is not suitable for doctest-style automated testing.

    Example (runs without errors):
    >>> #speak_text("Hello, this is a test.")  # plays audio
    """
    response = requests.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={
            "Authorization": f"Bearer " + api_key,
            "Content-Type": "application/json"
        },
        json={
            "model": "playai-tts",
            "input": text,
            "voice": "Briggs-PlayAI",
            "response_format":"wav"
        }
    )


    print("🔊 Playing TTS response...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(response.content)
        temp_path = temp_file.name

    try:
        # Load and play from temp file
        data, samplerate = sf.read(temp_path, dtype='float32')
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print("❌ Playback failed:", e)
    finally:
        # Clean up
        os.remove(temp_path)


def record_audio(filename="input.wav", duration=5, fs=44100):
    '''
    Records audio input and saves to a file.

    Example (not tested automatically):
    >>> # record_audio("test.wav", duration=2)
    '''
    print("🎙️ Recording... speak now")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print("✅ Recording saved.")

def transcribe_audio(filename="input.wav"):
    """
    Transcribes an audio file using Groq's Whisper API.

    Parameters:
        filename (str): Path to a WAV audio file.

    Returns:
        str: The transcribed text from the audio.

    Example (not tested by doctest):
    # >>> transcribe_audio("hello.wav")
    # 'Hello, this is a test recording.'
    """
    with open(filename, "rb") as f:
        response = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer " + api_key
            },
            files={
                "file": f
            },
            data={
                "model": "whisper-large-v3"
            }
        )
    try:
        json_data = response.json()
    except Exception as e:
        print("❌ Failed to parse response JSON:", e)
        print(response.text)
        return ""

    if "text" not in json_data:
        print("❌ Groq STT API did not return 'text'. Here's the full response:")
        print(json.dumps(json_data, indent=2))
        return ""

    return json_data["text"]

def voice_query_and_answer(doc_path_or_url="sample.txt",duration=5):
    '''
    Asks a voice question and answers based on document.

    Example (runs without errors):
    >>> # voice_query_and_answer("sample.txt", duration=5)
    '''
    try:
        document_text = load_text(doc_path_or_url)
        print("Document loaded.")
    except Exception as e:
        print("Failed to load document:",e)
        return
    
    input("Press Enter to record your question")
    record_audio("input.wav", duration = duration)

    query = transcribe_audio("input.wav")
    if not query:
        print("No transcription returned")
        return
    print("Transcribed question:", query)

    query = maybe_translate_query(query,document_text)

    chunks = find_relevant_chunks(document_text, query)
    answer = "\n".join(chunks)
    print("💡 Relevant answer:\n", answer)

    # Step 6: Speak the answer
    speak_text(answer)


def chat_with_document(doc_path_or_url,duration =5):
    '''
    Starts a chat session with the document.

    Example (runs without errors):
    >>> # chat_with_document("sample.txt", duration=5)
    '''

    try:
        document_text = load_text(doc_path_or_url)
        print("✅ Document loaded.")
    except Exception as e:
        print("❌ Failed to load document:", e)
        return

    # Step 2
    summary = "Summary not implemented yet."  # (Optional: you can use LLM to summarize)

    # Step 3: Setup chat history
    messages = []
    system_prompt = (
        "You are a helpful assistant that answers questions based on the given document.\n"
        "Here is a summary of the document:\n"
        f"{summary}\n"
        "Always answer based on the document unless clearly told otherwise."
    )
    messages.append({"role": "system", "content": system_prompt})

    # Step 4: Infinite loop to ask questions
    while True:
        print("🎤 Press Enter to start recording your question (or type 'exit' to quit).")
        user_input = input().strip().lower()

        if user_input == "exit":
            print("Goodbye!")
            break

        # Record and transcribe the question
        record_audio("input.wav", duration=5)
        user_query = transcribe_audio("input.wav")
        if not user_query.strip():
            print("⚠️ No speech detected. Try again.")
            continue

        print(f"📝 You asked: {user_query}")

        # If user says "exit", then break
        if user_query.lower().strip() in ["exit", "quit"]:
            print("👋 Exiting. Goodbye!")
            break

        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(document_text, user_query)
        context = "\n".join(relevant_chunks)

        # Step 5: Build new user message
        user_message = (
            f"Here are some parts of the document that might help:\n{context}\n\n"
            f"My question is: {user_query}"
        )
        messages.append({"role": "user", "content": user_message})

        # Step 6: Send to Groq
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 500
            }
        )

        try:
            assistant_reply = response.json()["choices"][0]["message"]["content"]
            print("💡 Answer:\n", assistant_reply)
            messages.append({"role": "assistant", "content": assistant_reply})
            speak_text(assistant_reply)
        except Exception as e:
            print("❌ Failed to get a valid response:", e)
            print(response.text)

print(find_relevant_chunks("The cat sat on the mat. The dog barked loudly.", "cat", num_chunks=1))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python docchat.py <file_or_url>")
        sys.exit(1)
    
    doc_path_or_url = sys.argv[1]
    print("🧠 Voice Q&A System Starting...")
    chat_with_document(doc_path_or_url, duration=5)