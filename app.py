# Install required libraries
import os
import streamlit as st
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import torchaudio

# Set up Streamlit page
st.title("Audio-to-Text Transcription with Qwen2-Audio-7B-Instruct")

# Load the processor and model
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    return processor, model

processor, model = load_model()

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, etc.)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file)

    # Load the audio file
    waveform, sample_rate = torchaudio.load(uploaded_file)

    # Resample to 16 kHz if needed
    resample_rate = 16000
    if sample_rate != resample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)
        waveform = resampler(waveform)

    # Process the audio with the model
    with st.spinner("Transcribing audio..."):
        inputs = processor(waveform, sampling_rate=resample_rate, return_tensors="pt")
        outputs = model.generate(**inputs)
        transcription = processor.decode(outputs[0], skip_special_tokens=True)

    # Display the transcription
    st.subheader("Transcription:")
    st.write(transcription)
