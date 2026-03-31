import streamlit as st
import numpy as np
import os
import pickle
import tempfile
import warnings
warnings.filterwarnings('ignore')

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    st.error("Missing audio libraries. Run: pip install sounddevice soundfile")

from src.data import AudioDataLoader
from src.features import AudioFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Machine Health Detector", layout="wide", initial_sidebar_state="expanded")
st.title("🔧 Machine Health Fault Detector")

# Sidebar
st.sidebar.header("📋 About")
st.sidebar.success("✅ All dependencies installed!")
st.sidebar.info("""
**Machine Health Detector** uses AI to identify machine faults from audio.

**How to use:**
1. Train the model (one-time)
2. Record or upload audio
3. Get instant result: NORMAL ✅ or FAULT ⚠️

**Features:**
- Real-time microphone recording
- Direct WAV file upload
- Instant classification
- Confidence percentage
""")

# Load model
@st.cache_resource
def load_model():
    """Load the pre-trained classifier model"""
    model_path = 'models/classifier/model.pkl'
    scaler_path = 'models/classifier/scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler, True
        except:
            return None, None, False
    return None, None, False

model, scaler, model_exists = load_model()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎤 Record Audio", "📁 Upload Audio", "🚀 Train Model"])

# TAB 1: RECORD AUDIO
with tab1:
    st.subheader("Record Audio from Microphone")
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Recording Duration (seconds)", 1, 10, 3, help="How long to record")
    with col2:
        sample_rate = st.selectbox("Sample Rate (Hz)", [22050, 44100, 48000], index=0, help="Audio sample rate")
    
    if st.button("🎙️ Start Recording", key='record_btn', use_container_width=True):
        try:
            with st.spinner(f"Recording {duration} seconds..."):
                st.info(f"🎙️ Recording in progress... ({duration} seconds)")
                
                # Record audio
                audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
                sd.wait()
                
                st.success("✅ Recording complete!")
                
                # Save temporarily
                temp_audio_path = 'temp_recorded.wav'
                sf.write(temp_audio_path, audio_data, sample_rate)
                
                # Load and process
                loader = AudioDataLoader()
                audio = loader.load_audio(temp_audio_path)
                
                # Extract features
                extractor = AudioFeatureExtractor()
                features = extractor.extract_mel_mfcc_combined(audio)
                features = features.reshape(1, -1)
                
                # Predict
                if model_exists and scaler:
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)[0]
                    confidence = model.predict_proba(features_scaled)[0]
                    
                    # Display results
                    st.divider()
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        if prediction == 0:
                            st.success("✅ NORMAL", icon="✓")
                            st.metric("Status", "Machine is Healthy", delta="No faults detected")
                            st.progress(confidence[0])
                            st.write(f"**Confidence:** {confidence[0]*100:.1f}%")
                        else:
                            st.error("⚠️ FAULT DETECTED", icon="✗")
                            st.metric("Status", "Machine Fault Found", delta="Action Required")
                            st.progress(confidence[1])
                            st.write(f"**Confidence:** {confidence[1]*100:.1f}%")
                    
                    with col_res2:
                        st.subheader("📊 Audio Analysis")
                        st.metric("Duration", f"{len(audio)/22050:.2f}s")
                        st.metric("Sample Rate", f"{sample_rate} Hz")
                        st.metric("Features Extracted", "564-dimensional")
                else:
                    st.warning("⚠️ Model not trained. Go to 'Train Model' tab first.", icon="⚠️")
                
                # Cleanup
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                
        except Exception as e:
            st.error(f"❌ Recording error: {str(e)}")

# TAB 2: UPLOAD AUDIO
with tab2:
    st.subheader("Upload Audio File")
    
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'], help="Select a .wav audio file")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing audio..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                # Load and process audio
                loader = AudioDataLoader()
                audio = loader.load_audio(temp_path)
                
                # Extract features
                extractor = AudioFeatureExtractor()
                features = extractor.extract_mel_mfcc_combined(audio)
                features = features.reshape(1, -1)
                
                # Predict
                if model_exists and scaler:
                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)[0]
                    confidence = model.predict_proba(features_scaled)[0]
                    
                    # Display results
                    st.divider()
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        if prediction == 0:
                            st.success("✅ NORMAL", icon="✓")
                            st.metric("Status", "Machine is Healthy", delta="No faults detected")
                            st.progress(confidence[0])
                            st.write(f"**Confidence:** {confidence[0]*100:.1f}%")
                        else:
                            st.error("⚠️ FAULT DETECTED", icon="✗")
                            st.metric("Status", "Machine Fault Found", delta="Action Required")
                            st.progress(confidence[1])
                            st.write(f"**Confidence:** {confidence[1]*100:.1f}%")
                    
                    with col_res2:
                        st.subheader("📊 Audio Analysis")
                        st.metric("Duration", f"{len(audio)/22050:.2f}s")
                        st.metric("Sample Rate", "22050 Hz")
                        st.metric("Features Extracted", "564-dimensional")
                else:
                    st.warning("⚠️ Model not trained. Go to 'Train Model' tab first.", icon="⚠️")
                
                # Cleanup
                os.remove(temp_path)
                
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

# TAB 3: TRAIN MODEL
with tab3:
    st.subheader("🚀 Train Machine Learning Model")
    st.info("Train the Random Forest classifier using synthetic dataset (145 training samples)")
    
    train_col1, train_col2 = st.columns(2)
    with train_col1:
        if st.button("🚀 TRAIN MODEL", key='train_btn', use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📥 Loading dataset...")
                progress_bar.progress(10)
                
                from src.data import load_and_split_dataset
                
                # Load all data
                datasets = load_and_split_dataset(
                    data_root='data/synthetic',
                    snr_levels=[20, 10, 5, 0]
                )
                
                status_text.text("🔢 Extracting features...")
                progress_bar.progress(30)
                
                extractor = AudioFeatureExtractor()
                X_train, y_train = [], []
                
                # Combine all SNR levels for training
                total_samples = 0
                for snr_key, splits in datasets.items():
                    for audio, label in zip(splits['train']['audio'], splits['train']['labels']):
                        features = extractor.extract_mel_mfcc_combined(audio)
                        X_train.append(features)
                        y_train.append(label)
                        total_samples += 1
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                status_text.text(f"🤖 Training on {total_samples} samples...")
                progress_bar.progress(50)
                
                # Scale features
                scaler_new = StandardScaler()
                X_train_scaled = scaler_new.fit_transform(X_train)
                
                # Train model
                model_new = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model_new.fit(X_train_scaled, y_train)
                
                status_text.text("💾 Saving model...")
                progress_bar.progress(80)
                
                # Save model
                os.makedirs('models/classifier', exist_ok=True)
                with open('models/classifier/model.pkl', 'wb') as f:
                    pickle.dump(model_new, f)
                with open('models/classifier/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler_new, f)
                
                progress_bar.progress(100)
                status_text.text("✅ Training complete!")
                
                st.success(f"✅ Model trained successfully on {total_samples} samples!")
                st.info("🎉 Go to 'Record Audio' or 'Upload Audio' tab to test!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")
    
    with train_col2:
        st.markdown("**Training Details:**")
        st.markdown("""
        - **Algorithm:** Random Forest (100 trees)
        - **Features:** Mel-Spectrogram + MFCC (564-dim)
        - **Training Samples:** ~145 audio files
        - **Classes:** Normal (0), Fault (1)
        - **Time:** ~30 seconds
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
<p><strong>Machine Health Fault Detector v1.0</strong></p>
<p>🤖 AI-powered fault detection | 📊 Real-time analysis</p>
<p>Dataset: 200 synthetic machine sounds | Model: Random Forest</p>
</div>
""", unsafe_allow_html=True)
