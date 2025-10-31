import streamlit as st
import requests
import base64
import io
import numpy as np
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
import openai
from faster_whisper import WhisperModel
import wave

# Page config
st.set_page_config(
    page_title="Clinic Voice Assistant Demo",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Clinic Voice Assistant Demo")
st.markdown("*Talk to our AI assistant to schedule appointments and get information*")

# Configuration
TTS_SERVER_URL = "https://6ldo5kjjcjvh5j-8000.proxy.runpod.net"
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": """You are a helpful voice assistant for a medical clinic. 
        You help patients schedule appointments, answer questions about services, and provide general assistance.
        Keep responses concise and conversational since they will be spoken aloud.
        Be friendly and professional."""}
    ]

if 'last_audio_hash' not in st.session_state:
    st.session_state.last_audio_hash = None

if 'processing_audio' not in st.session_state:
    st.session_state.processing_audio = False

if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None

if 'current_audio_message' not in st.session_state:
    st.session_state.current_audio_message = None

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

conversation_history = st.session_state.conversation_history

@st.cache_resource
def load_whisper():
    """Load Whisper model once and cache it."""
    return WhisperModel("small.en", device="cpu", compute_type="int8")

def is_audio_significant(audio_bytes):
    """Check if audio contains significant content (not just silence/noise)."""
    try:
        # Convert bytes to numpy array for analysis
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
        
        try:
            # Read the audio file
            with wave.open(temp_filename, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Calculate audio properties
            if len(audio_data) == 0:
                return False
            
            # Check duration (minimum 0.5 seconds)
            duration = len(audio_data) / sample_rate
            if duration < 0.5:
                return False
            
            # Check volume/energy level
            rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))
            # Threshold for significant audio (adjust as needed)
            if rms < 500:  # Very quiet audio threshold
                return False
            
            # Check for sustained audio (not just a brief spike)
            window_size = int(sample_rate * 0.1)  # 100ms windows
            windows = [audio_data[i:i+window_size] for i in range(0, len(audio_data), window_size)]
            significant_windows = sum(1 for window in windows if len(window) > 0 and np.sqrt(np.mean(window.astype(float) ** 2)) > 300)
            
            # At least 30% of windows should have significant audio
            if significant_windows / len(windows) < 0.3:
                return False
            
            return True
            
        finally:
            os.unlink(temp_filename)
            
    except Exception as e:
        st.error(f"Audio analysis error: {e}")
        return False

def transcribe_audio(audio_bytes):
    """Transcribe audio to text."""
    try:
        # First check if audio is significant enough to process
        if not is_audio_significant(audio_bytes):
            return None
        
        whisper_model = load_whisper()
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
        
        try:
            # Transcribe with VAD (Voice Activity Detection)
            segments, info = whisper_model.transcribe(
                temp_filename, 
                beam_size=5, 
                language="en",
                vad_filter=True,  # Enable voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)  # Require 500ms silence
            )
            text = " ".join([segment.text.strip() for segment in segments])
            return text.strip() if text.strip() else None
        finally:
            os.unlink(temp_filename)
            
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def get_openai_response(user_text):
    """Get response from OpenAI."""
    try:
        # Add user message to conversation
        conversation_history.append({"role": "user", "content": user_text})
        
        # Get response from OpenAI
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            max_tokens=150,
            temperature=0.7
        )
        
        assistant_text = response.choices[0].message.content.strip()
        
        # Add assistant response to conversation
        conversation_history.append({"role": "assistant", "content": assistant_text})
        st.session_state.conversation_history = conversation_history
        
        return assistant_text
        
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "I'm sorry, I'm having trouble processing your request right now."

def synthesize_speech(text):
    """Convert text to speech using the TTS server."""
    try:
        response = requests.post(
            f"{TTS_SERVER_URL}/synthesize",
            json={"text": text, "max_new_tokens": 1200, "temperature": 0.6},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data['audio_base64']
            else:
                st.error(f"TTS synthesis failed: {data.get('error')}")
                return None
        else:
            st.error(f"TTS server error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"TTS synthesis error: {e}")
        return None

def check_tts_server():
    """Check if TTS server is available."""
    try:
        response = requests.get(f"{TTS_SERVER_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('model_loaded', False)
        return False
    except:
        return False

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎤 Voice Input")
    
    # Check server status
    server_status = check_tts_server()
    if server_status:
        st.success("✅ Voice AI Server Online")
    else:
        st.error("❌ Voice AI Server Offline")
        st.stop()
    
    # Manual recording toggle
    col_mic1, col_mic2 = st.columns([1, 3])
    
    with col_mic1:
        if not st.session_state.is_recording:
            if st.button("🎤 Start Recording", type="primary"):
                st.session_state.is_recording = True
                st.session_state.recorded_audio = None
                st.rerun()
        else:
            if st.button("🛑 Stop Recording", type="secondary"):
                st.session_state.is_recording = False
                st.rerun()
    
    with col_mic2:
        if st.session_state.is_recording:
            st.markdown("**🔴 RECORDING... Click 'Stop Recording' when done**")
        else:
            st.markdown("**⚪ Click 'Start Recording' to begin**")
    
    # Audio input when recording
    if st.session_state.is_recording:
        audio_bytes = st.audio_input("Recording in progress...", key="voice_input")
        if audio_bytes:
            st.session_state.recorded_audio = audio_bytes
            st.session_state.is_recording = False
            st.success("✅ Recording captured! Processing...")
            st.rerun()
    
    # Process recorded audio
    if st.session_state.recorded_audio and not st.session_state.processing_audio:
        audio_bytes = st.session_state.recorded_audio
        st.session_state.recorded_audio = None  # Clear after processing
    
    # Manual text input as fallback
    st.subheader("✏️ Text Input (Alternative)")
    manual_text = st.text_area("Or type your message here:")
    
    if st.button("Send Text Message") and manual_text:
        # Process manual text input
        with st.spinner("Getting AI response..."):
            ai_response = get_openai_response(manual_text)
        
        with st.spinner("Generating speech..."):
            audio_base64 = synthesize_speech(ai_response)
        
        if audio_base64:
            # Store audio in session state for persistence
            audio_data = base64.b64decode(audio_base64)
            st.session_state.current_audio = audio_data
            st.session_state.current_audio_message = ai_response
            st.audio(audio_data, format="audio/wav")
        
        # Add to conversation display
        st.session_state.conversation_history = conversation_history

with col2:
    st.subheader("💬 Conversation")
    
    # Display conversation history
    for i, message in enumerate(conversation_history[1:], 1):  # Skip system message
        if message["role"] == "user":
            st.markdown(f"**🗣️ You:** {message['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {message['content']}")
    
    # Display current audio if available
    if st.session_state.current_audio is not None:
        st.markdown("---")
        st.markdown("**🔊 Latest Response:**")
        if st.session_state.current_audio_message:
            st.markdown(f"*{st.session_state.current_audio_message}*")
        st.audio(st.session_state.current_audio, format="audio/wav")
    
    # Process audio input with duplicate prevention
    if audio_bytes and not st.session_state.processing_audio:
        # Create hash of audio to prevent reprocessing
        import hashlib
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            st.session_state.processing_audio = True
            
            st.subheader("🔄 Processing Your Message")
            
            with st.spinner("Analyzing audio..."):
                user_text = transcribe_audio(audio_bytes)
            
            if user_text and len(user_text.strip()) > 5:  # Require at least 5 characters
                st.success(f"📝 You said: *{user_text}*")
                
                with st.spinner("Getting AI response..."):
                    ai_response = get_openai_response(user_text)
                
                st.success(f"🤖 Assistant: *{ai_response}*")
                
                with st.spinner("Generating speech..."):
                    audio_base64 = synthesize_speech(ai_response)
                
                if audio_base64:
                    # Store audio in session state
                    audio_data = base64.b64decode(audio_base64)
                    st.session_state.current_audio = audio_data
                    st.session_state.current_audio_message = ai_response
                    
                    st.success("🔊 Audio response generated! Check the conversation section below.")
                
                # Don't rerun immediately - let user interact with audio
                st.session_state.processing_audio = False
            else:
                st.warning("❌ No clear speech detected or speech too short. Please try again.")
                st.session_state.processing_audio = False

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This is a demo of our AI-powered clinic voice assistant.
    
    **Features:**
    - 🎤 Voice input and output
    - 🤖 AI-powered responses
    - 📅 Appointment scheduling help
    - ❓ General clinic information
    
    **How to use:**
    1. Click 'Start Recording' to begin
    2. Speak your question or request
    3. Click 'Stop Recording' when finished
    4. Wait for the AI response
    5. Listen to the spoken answer in the conversation
    
    **Example questions:**
    - "I'd like to schedule an appointment"
    - "What services do you offer?"
    - "What are your hours?"
    """)
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = [conversation_history[0]]  # Keep system message
        st.session_state.last_audio_hash = None
        st.session_state.processing_audio = False
        st.session_state.current_audio = None
        st.session_state.current_audio_message = None
        st.session_state.is_recording = False
        st.session_state.recorded_audio = None
        st.rerun()
    
    if st.button("🔄 Reset Audio Processing"):
        st.session_state.last_audio_hash = None
        st.session_state.processing_audio = False
        st.session_state.current_audio = None
        st.session_state.current_audio_message = None
        st.session_state.is_recording = False
        st.session_state.recorded_audio = None
        st.success("Audio processing reset!")
        st.rerun()