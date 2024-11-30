# Import necessary libraries
import streamlit as st
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
import torchaudio

# Streamlit app title
st.title("Audio-to-Text Transcription with Qwen2-Audio-7B-Instruct")

# Load the processor and model
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    return processor, model

processor, model = load_model()

# File uploader for audio files
uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, FLAC)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Display an audio player in the Streamlit app
    st.audio(uploaded_file)

    # Load the uploaded audio file using torchaudio
    waveform, sample_rate = torchaudio.load(uploaded_file)

    # Resample audio to 16 kHz if the sample rate is different
    resample_rate = 16000
    if sample_rate != resample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)
        waveform = resampler(waveform)

    # Process the audio with the model
    with st.spinner("Transcribing audio..."):
        inputs = processor(waveform, sampling_rate=resample_rate, return_tensors="pt")
        outputs = model.generate(**inputs)
        transcription = processor.decode(outputs[0], skip_special_tokens=True)

    # Display the transcription result
    st.subheader("Transcription:")
    st.write(transcription)
else:
    st.info("Please upload an audio file to start transcription.")

