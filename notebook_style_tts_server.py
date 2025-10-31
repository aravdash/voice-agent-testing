#!/usr/bin/env python3
"""
TTS Server matching the exact notebook inference process.
Uses the same token structure and code processing as the training notebook.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import io
import wave
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotebookStyleTTSServer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio parameters
        self.sample_rate = 24000
        self.codec_name = "snac_24khz"
        
        # Special tokens from notebook
        self.START_TOKEN = 128259  # Start of human
        self.END_OF_TEXT = 128009  # End of text
        self.END_OF_HUMAN = 128260  # End of human
        self.PAD_TOKEN = 128263    # Padding token
        self.MARKER_TOKEN = 128257  # Token to find for cropping
        self.REMOVE_TOKEN = 128258  # Token to remove
        self.EOS_TOKEN = 128258    # EOS token for generation
        self.CODE_OFFSET = 128266  # Offset for audio codes
        
    def load_model(self):
        """Load the merged model from Hugging Face."""
        try:
            logger.info("Loading merged Orpheus model from Hugging Face...")
            
            # Load the merged model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    "aravdash/orpheus-voice",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
            except ImportError:
                logger.warning("FlashAttention2 not available, using standard attention")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "aravdash/orpheus-voice",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "aravdash/orpheus-voice",
                trust_remote_code=True
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load SNAC model
            logger.info("Loading SNAC model...")
            from snac import SNAC
            self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            # Keep SNAC on CPU as in notebook
            self.snac_model.to("cpu")
            
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"Model dtype: {self.model.dtype}")
            logger.info(f"Model device: {self.model.device}")
            logger.info(f"Vocabulary size: {len(self.tokenizer)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            return False
    
    def prepare_input(self, prompt, chosen_voice=None):
        """Prepare input exactly as in notebook."""
        # Add voice prefix if specified
        if chosen_voice:
            prompt = f"{chosen_voice}: {prompt}"
        
        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Add special tokens: SOH + prompt + EOT + EOH
        start_token = torch.tensor([[self.START_TOKEN]], dtype=torch.int64)
        end_tokens = torch.tensor([[self.END_OF_TEXT, self.END_OF_HUMAN]], dtype=torch.int64)
        
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones_like(modified_input_ids)
        
        return modified_input_ids.to(self.device), attention_mask.to(self.device)
    
    def redistribute_codes(self, code_list):
        """Redistribute codes exactly as in notebook."""
        layer_1 = []
        layer_2 = []
        layer_3 = []
        
        for i in range((len(code_list)+1)//7):
            if 7*i < len(code_list):
                layer_1.append(code_list[7*i])
            if 7*i+1 < len(code_list):
                layer_2.append(code_list[7*i+1]-4096)
            if 7*i+2 < len(code_list):
                layer_3.append(code_list[7*i+2]-(2*4096))
            if 7*i+3 < len(code_list):
                layer_3.append(code_list[7*i+3]-(3*4096))
            if 7*i+4 < len(code_list):
                layer_2.append(code_list[7*i+4]-(4*4096))
            if 7*i+5 < len(code_list):
                layer_3.append(code_list[7*i+5]-(5*4096))
            if 7*i+6 < len(code_list):
                layer_3.append(code_list[7*i+6]-(6*4096))
        
        codes = [
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]
        
        # Decode with SNAC (on CPU as in notebook)
        audio_hat = self.snac_model.decode(codes)
        return audio_hat
    
    def synthesize(self, text, max_new_tokens=1200, temperature=0.6, chosen_voice=None):
        """Synthesize speech exactly as in notebook."""
        try:
            logger.info(f"Synthesizing: '{text}'")
            
            # Prepare input
            input_ids, attention_mask = self.prepare_input(text, chosen_voice)
            input_length = input_ids.shape[1]
            logger.info(f"Input length: {input_length} tokens")
            
            # Generate exactly as in notebook
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    num_return_sequences=1,
                    eos_token_id=self.EOS_TOKEN,
                    use_cache=True
                )
            
            logger.info(f"Generated {generated_ids.shape[1]} total tokens")
            
            # Process tokens exactly as in notebook
            token_to_find = self.MARKER_TOKEN
            token_to_remove = self.REMOVE_TOKEN
            
            # Find marker token
            token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
            
            if len(token_indices[1]) > 0:
                last_occurrence_idx = token_indices[1][-1].item()
                cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
                logger.info(f"Found marker token at position {last_occurrence_idx}")
            else:
                cropped_tensor = generated_ids
                logger.warning("No marker token found, using full generated sequence")
            
            # Remove unwanted tokens
            mask = cropped_tensor != token_to_remove
            processed_rows = []
            
            for row in cropped_tensor:
                masked_row = row[row != token_to_remove]
                processed_rows.append(masked_row)
            
            # Convert to code lists
            code_lists = []
            for row in processed_rows:
                row_length = row.size(0)
                new_length = (row_length // 7) * 7
                trimmed_row = row[:new_length]
                # Subtract code offset
                trimmed_row = [t.item() - self.CODE_OFFSET for t in trimmed_row]
                code_lists.append(trimmed_row)
            
            logger.info(f"Extracted {len(code_lists)} code sequences")
            
            if not code_lists or len(code_lists[0]) == 0:
                return {
                    "success": False,
                    "error": "No audio codes extracted",
                    "debug_info": {
                        "generated_tokens": generated_ids.shape[1],
                        "cropped_tokens": cropped_tensor.shape[1] if len(cropped_tensor.shape) > 1 else 0,
                        "found_marker": len(token_indices[1]) > 0,
                        "code_lists_count": len(code_lists)
                    }
                }
            
            # Generate audio from first code list
            audio_samples = self.redistribute_codes(code_lists[0])
            
            # Convert to numpy
            if isinstance(audio_samples, torch.Tensor):
                audio = audio_samples.detach().squeeze().to("cpu").numpy()
            else:
                audio = audio_samples
            
            # Ensure it's 1D and normalize
            if audio.ndim > 1:
                audio = audio.flatten()
            audio = np.clip(audio, -1.0, 1.0)
            
            logger.info(f"Generated audio: {len(audio)} samples at {self.sample_rate}Hz")
            
            return {
                "success": True,
                "audio": audio,
                "sample_rate": self.sample_rate,
                "message": f"Generated {len(audio)} audio samples from {len(code_lists[0])} codes"
            }
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

# Flask app
app = Flask(__name__)
CORS(app)

# Global TTS server instance
tts_server = NotebookStyleTTSServer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if tts_server.model is not None else "loading",
        "model_loaded": tts_server.model is not None,
        "device": str(tts_server.device)
    })

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """Synthesize speech from text."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "Missing 'text' field"}), 400
        
        text = data['text']
        max_tokens = data.get('max_new_tokens', 1200)
        temperature = data.get('temperature', 0.6)
        chosen_voice = data.get('chosen_voice', None)
        
        # Synthesize
        result = tts_server.synthesize(text, max_tokens, temperature, chosen_voice)
        
        if result['success']:
            # Convert audio to base64 for JSON response
            audio = result['audio']
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(result['sample_rate'])
                wav_file.writeframes(audio_int16.tobytes())
            
            # Get WAV bytes and encode to base64
            wav_bytes = wav_buffer.getvalue()
            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            return jsonify({
                "success": True,
                "audio_base64": audio_base64,
                "sample_rate": result['sample_rate'],
                "duration": len(audio) / result['sample_rate'],
                "message": result['message']
            })
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in synthesize endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information."""
    if tts_server.model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_name": "aravdash/orpheus-voice",
        "model_type": "merged",
        "vocab_size": len(tts_server.tokenizer),
        "device": str(tts_server.device),
        "sample_rate": tts_server.sample_rate,
        "codec": tts_server.codec_name
    })

if __name__ == "__main__":
    logger.info("Starting notebook-style TTS server...")
    
    # Load the model
    if tts_server.load_model():
        logger.info("🚀 TTS server ready!")
        app.run(host="0.0.0.0", port=8000, debug=False)  # Back to port 8000
    else:
        logger.error("❌ Failed to start server - model loading failed")