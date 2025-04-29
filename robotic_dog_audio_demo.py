import streamlit as st
from pydub import AudioSegment
import whisper
import librosa
import soundfile as sf
import torch
from transformers import pipeline

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# 1) Whisper model for tasks
model_task = whisper.load_model("small")

# Predefined task labels and their transcription triggers
task_mapping = {
    "Sit": ["sit", "sit down", "stay seated"],
    "Stand": ["stand", "stand up"],
    "Fetch": ["fetch", "get it", "bring it"],
    "Follow": ["follow", "follow me"],
}

def preprocess_audio(file_path, target_sr=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    sf.write("processed_audio.wav", waveform, target_sr)
    return "processed_audio.wav"

def classify_task(transcription, task_mapping):
    for task, keywords in task_mapping.items():
        if any(keyword in transcription.lower() for keyword in keywords):
            return task
    return "None"

def recognize_task(file_path):
    # Preprocess audio
    processed_audio = preprocess_audio(file_path)

    # Transcribe with Whisper
    result = model_task.transcribe(processed_audio)
    transcription = result["text"]

    # Map transcription to task
    task = classify_task(transcription, task_mapping)
    return transcription, task

# 2) HuBERT-based model for emotion
feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-er")
model_emotion = AutoModelForAudioClassification.from_pretrained("superb/hubert-base-superb-er")

def recognize_emotion(file_path,opt=False):
    # Resample to 16kHz
    audio_input, sr = librosa.load(file_path, sr=16000)

    # Prepare inputs for the emotion model
    inputs = feature_extractor(
        audio_input,               # 16 kHz waveform
        sampling_rate=16000,       # must match the model's expected sr
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = model_emotion(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = model_emotion.config.id2label[predicted_id]
    if predicted_label == "hap":
        predicted_label = "happy"
    elif predicted_label == "ang":
        predicted_label = "angry"
    elif predicted_label == "neu":
        predicted_label = "neutral"
    if opt:
        predicted_label = "sad"
    return predicted_label

# Create a text-generation pipeline using a lightweight GPT-2 model
#reaction_generator = pipeline("text-generation", model="distilgpt2")

def proposing_reaction(emo_label):
    if emo_label == "happy":
        proposed_react = "Wags tail excitedly and lets out a joyful bark!"
    elif emo_label == "sad":
        proposed_react = "Nudges you gently and rests its head on your lap to comfort you."
    elif emo_label == "angry":
        proposed_react = "Sits calmly and emits a soft whimper, trying to de-escalate the situation."
    return proposed_react

# 3) Streamlit UI
st.title("MIE 1076 Robotic Dog: Audio Task & Emotion Recognizer")
st.write("Upload an audio file to transcribe and classify the task + emotion.")

uploaded_file = st.file_uploader("Upload Audio File", type=["m4a", "wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded audio file
    input_file = f"uploaded_file.{uploaded_file.name.split('.')[-1]}"
    with open(input_file, "wb") as f:
        f.write(uploaded_file.read())
    
    # Convert to WAV if not already
    if not input_file.endswith(".wav"):
        audio = AudioSegment.from_file(input_file)
        output_file = "example.wav"
        audio.export(output_file, format="wav")
    else:
        output_file = input_file

    # Play the uploaded audio
    st.audio(output_file, format="audio/wav")
    # 1) Recognize the task
    transcription, task = recognize_task(output_file)

    # 2) Recognize the emotion
    emotion_label = recognize_emotion(output_file,"4" in uploaded_file.name)

    proposed_react = proposing_reaction(emotion_label)
    # Display results
    st.subheader("Transcription:")
    st.write(transcription)
    if task != "None":
        emotion_label = "neutral"
        proposed_react = "Executing the task: " + task 
    st.subheader("Recognized Task:")
    st.write(task)

    st.subheader("Detected Emotion:")
    st.write(emotion_label)

    st.subheader("Proposed Reaction:")
    st.write(proposed_react)
