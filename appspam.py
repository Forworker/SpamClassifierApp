import os
import tempfile
import streamlit as st
import pickle
import re
import whisper
import nltk

# Add local nltk_data path
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

from nltk.corpus import stopwords

# ==================== Load Trained Components ====================
with open("model_nb.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load Whisper model
asr_model = whisper.load_model("base")

# ==================== Text Preprocessing ====================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    stop_words = set(stopwords.words("english"))
    return ' '.join([word for word in text.split() if word not in stop_words])

# ==================== Simple Whisper Transcription ====================
def transcribe_audio(filename):
    result = asr_model.transcribe(filename)
    return result["text"]

# ==================== App UI ====================
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("Spam Classifier from Text & Audio")
st.markdown("Enter text or upload an audio clip to determine if the message is Spam.")

# ========== Text Input ==========
st.subheader("Manual text entry ")
text_input = st.text_area("Write your message here:")

if text_input:
    cleaned = preprocess_text(text_input)
    vect_text = vectorizer.transform([cleaned])
    pred = model.predict(vect_text)[0]
    st.success("🟠 SPAM" if pred else "🟢 NOT SPAM")

# ========== Audio Upload ==========
st.subheader("Or upload an audio clip 🎙️ ")
audio_file = st.file_uploader("Upload an audio file [mp3/wav/m4a]", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        temp_filename = tmp.name

    st.info("⏳ Converting audio to text...")
    transcribed_text = transcribe_audio(temp_filename)
    st.text_area("🗣️ Text extracted from the audio:", transcribed_text)

    cleaned_audio_text = preprocess_text(transcribed_text)
    vect_audio = vectorizer.transform([cleaned_audio_text])
    pred_audio = model.predict(vect_audio)[0]
    st.success("🟠 SPAM" if pred_audio else "🟢 NOT SPAM")




    
