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
if 'conversation_history' in st.session_state:
    conversation_history = st.session_state.conversation_history
else:
    conversation_history = [
        {"role": "system", "content": """You are a helpful voice assistant for a medical clinic. 
        You help patients schedule appointments, answer questions about services, and provide general assistance.
        Keep responses concise and conversational since they will be spoken aloud.
        Be friendly and professional."""}
    ]
    st.session_state.conversation_history = conversation_history

@st.cache_resource
def load_whisper():
    """Load Whisper model once and cache it."""
    return WhisperModel("small.en", device="cpu", compute_type="int8")

def transcribe_audio(audio_bytes):
    """Transcribe audio to text."""
    try:
        whisper_model = load_whisper()
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
        
        try:
            # Transcribe
            segments, info = whisper_model.transcribe(temp_filename, beam_size=5, language="en")
            text = " ".join([segment.text.strip() for segment in segments])
            return text.strip()
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
    
    # Audio recorder
    audio_bytes = audio_recorder(
        text="Click to record your message",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="2x"
    )
    
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
            # Decode and play audio
            audio_data = base64.b64decode(audio_base64)
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
    
    # Process audio input
    if audio_bytes:
        st.subheader("🔄 Processing Your Message")
        
        with st.spinner("Transcribing audio..."):
            user_text = transcribe_audio(audio_bytes)
        
        if user_text and len(user_text.strip()) > 3:
            st.success(f"📝 You said: *{user_text}*")
            
            with st.spinner("Getting AI response..."):
                ai_response = get_openai_response(user_text)
            
            st.success(f"🤖 Assistant: *{ai_response}*")
            
            with st.spinner("Generating speech..."):
                audio_base64 = synthesize_speech(ai_response)
            
            if audio_base64:
                st.success("🔊 Click play to hear the response:")
                # Decode and play audio
                audio_data = base64.b64decode(audio_base64)
                st.audio(audio_data, format="audio/wav")
            
            # Update conversation display
            st.rerun()
        else:
            st.warning("❌ No clear speech detected. Please try again.")

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
    1. Click the microphone button
    2. Speak your question or request
    3. Wait for the AI response
    4. Listen to the spoken answer
    
    **Example questions:**
    - "I'd like to schedule an appointment"
    - "What services do you offer?"
    - "What are your hours?"
    """)
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = [conversation_history[0]]  # Keep system message
        st.rerun()