import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import streamlit as st
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

# --- Configuration ---
MODEL_NAME = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
SAMPLE_RATE = 16_000

# --- Model Architecture ---
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, x):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        states = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
        if attention_mask is None:
            pooled = states.mean(dim=1)
        else:
            feature_mask = self._get_feature_vector_attention_mask(states.shape[1], attention_mask)
            weights = feature_mask.unsqueeze(-1).to(states.dtype)
            pooled = (states * weights).sum(1) / weights.sum(1).clamp_min(1)
        return pooled, self.age(pooled), torch.softmax(self.gender(pooled), dim=1)

@st.cache_resource
def load_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = AgeGenderModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device).eval()
    return processor, model, device

# --- Defensive Quality Checks & Preprocessing ---
def check_audio_quality(y, sr, min_duration=2.0, min_rms=0.005, max_clipping_ratio=0.05):
    quality_warnings = []
    duration = len(y) / sr
    if duration < min_duration:
        quality_warnings.append(f"Short audio: {duration:.2f}s (minimum recommended is {min_duration}s).")

    rms = np.sqrt(np.mean(y**2))
    if rms < min_rms:
        quality_warnings.append(f"Low audio volume (RMS: {rms:.5f}).")

    clipping_ratio = np.mean(np.abs(y) >= 0.99)
    if clipping_ratio > max_clipping_ratio:
        quality_warnings.append(f"High audio clipping ({clipping_ratio*100:.1f}% samples clipped).")
        
    return quality_warnings

def preprocess_audio(y, sr=SAMPLE_RATE):
    quality_warnings = check_audio_quality(y, sr)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = 0.95 * (y / peak)
    return y, quality_warnings

# --- Single Source Inference ---
def predict_raw(y, model, processor, device):
    inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
    with torch.inference_mode():
        _, age_logits, gender_probs = model(inputs.input_values)

    clamped_age = torch.clamp(age_logits.squeeze() * 100.0, 0.0, 100.0).item()

    gender_labels = ['child', 'female', 'male']
    gender_scores = gender_probs[0].cpu().numpy()
    predicted_gender = gender_labels[np.argmax(gender_scores)]
    gender_confidence = float(np.max(gender_scores))

    return clamped_age, predicted_gender, gender_confidence

def robust_predict(y, model, processor, device, sr=SAMPLE_RATE, num_segments=3):
    y, quality_warnings = preprocess_audio(y, sr)
    overall_age, overall_gender, gender_conf = predict_raw(y, model, processor, device)

    segment_ages = []
    if len(y) >= sr * 3:
        segments = np.array_split(y, num_segments)
        for seg in segments:
            seg_age, _, _ = predict_raw(seg, model, processor, device)
            segment_ages.append(seg_age)

    return {
        'predicted_age': overall_age,
        'gender': overall_gender,
        'gender_confidence': gender_conf,
        'segment_ages': segment_ages,
        'segment_spread': float(np.std(segment_ages)) if len(segment_ages) > 1 else 0.0,
        'warnings': quality_warnings
    }

# --- Streamlit UI ---
st.set_page_config(page_title="Voice Age & Gender Predictor", page_icon="🎙️")
st.title("🎙️ Voice Age & Gender Predictor")

processor, model, device = load_pipeline()

# Prompt script for live reading
with st.expander("📄 Suggested Reading Script", expanded=False):
    st.write("""
    *Good morning. Today I am participating in a voice-based age prediction experiment. 
    I will read this paragraph clearly and naturally at a comfortable speaking speed. 
    The system will analyze characteristics of my voice and estimate an age from this recording.*
    """)

tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload File"])

audio_source = None

with tab1:
    recorded_audio = st.audio_input("Record your voice")
    if recorded_audio:
        audio_source = recorded_audio

with tab2:
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "webm"])
    if uploaded_file:
        audio_source = uploaded_file

if audio_source is not None:
    st.audio(audio_source)
    
    if st.button("Predict Age & Gender", type="primary"):
        with st.spinner("Analyzing voice patterns..."):
            y, sr = librosa.load(audio_source, sr=SAMPLE_RATE, mono=True)
            res = robust_predict(y, model, processor, device, sr)

            if res['warnings']:
                for w in res['warnings']:
                    st.warning(f"⚠️ {w}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated Age", f"{res['predicted_age']:.1f} years")
            with col2:
                st.metric()

            if res['segment_ages']:
                st.caption(f"Segment Consistency (Std Dev): ±{res['segment_spread']:.2f} years")
