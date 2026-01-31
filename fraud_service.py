import torch
import librosa
import numpy as np
import streamlit as st
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os

# Configuration
MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"
TARGET_SR = 16000

@st.cache_resource
def load_voice_model():
    """Loads the pre-trained deepfake detection model. Cached for performance."""
    try:
        print(f"Loading model: {MODEL_NAME}...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model loaded on {device}")
        return feature_extractor, model, device
    except Exception as e:
        st.error(f"Failed to load model logic: {str(e)}")
        return None, None, None

def analyze_voice_spoof(audio_input) -> float:
    """
    Analyzes an audio file for deepfake signatures.
    
    Args:
        audio_input: Path to file (str) or file-like object (UploadedFile)
        
    Returns:
        float: Probability of being FAKE (0.0 to 1.0)
    """
    feature_extractor, model, device = load_voice_model()
    
    if not model:
        return 0.0 # Fail safe

    try:
        # Load and resample audio
        # librosa.load can handle both paths and file objects (with soundfile)
        y, sr = librosa.load(audio_input, sr=TARGET_SR)
        
        # Preprocess
        inputs = feature_extractor(y, return_tensors="pt", sampling_rate=TARGET_SR, padding=True)
        inputs = inputs.to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # DEBUG: Print everything to understand what the model is doing
        id2label = model.config.id2label
        print(f"DEBUG: id2label: {id2label}")
        print(f"DEBUG: Raw Logits: {logits}")
        print(f"DEBUG: Probs: {probs}")
        
        # Robust label finding
        spoof_idx = 1 # Default
        for idx, label in id2label.items():
            if str(label).lower() in ['spoof', 'fake', 'synthetic']:
                spoof_idx = int(idx)
                break
                
        fake_prob = probs[0][spoof_idx].item()
        
        print(f"DEBUG: Using Index {spoof_idx} for FAKE probability")
        print(f"DEBUG: Final Fake Prob: {fake_prob:.4f}")
        
        return fake_prob

    except Exception as e:
        st.error(f"Error analyzing audio: {str(e)}")
        return 0.0
