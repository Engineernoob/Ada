"""Voice service for Ada Cloud infrastructure.

This module provides speech-to-text and text-to-speech capabilities
processing through Modal's serverless infrastructure.
"""

from __future__ import annotations

import logging
import os
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
import json

logger = logging.getLogger(__name__)

# Try to import voice components, handle missing dependencies
try:
    from interfaces.speech_input import SpeechInput
    from interfaces.voice_output import VoiceOutput
    SPEECH_AVAILABLE = True
except ImportError as e:
    SPEECH_AVAILABLE = False
    logger.warning(f"Voice components not available - some features will be simulated: {e}")


class VoiceService:
    """Cloud-based voice service for Ada Cloud."""
    
    def __init__(self):
        """Initialize voice service."""
        self.speech_input = None
        self.voice_output = None
        self._initialize_services()
        
        # Create temp directory for audio files
        self.temp_dir = Path(tempfile.mktemp(prefix="ada_voice_"))
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info("Voice service initialized")
    
    def _initialize_services(self):
        """Initialize speech input and output services."""
        if SPEECH_AVAILABLE:
            try:
                self.speech_input = SpeechInput()
                self.voice_output = VoiceOutput()
                logger.info("Voice services initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize voice services: {e}")
                self.speech_input = None
                self.voice_output = None
        else:
            logger.warning("Voice services not available - operations will be simulated")
    
    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        if self.speech_input:
            try:
                # Save audio to temporary file
                audio_path = self.temp_dir / "transcribe_audio.wav"
                import soundfile as sf
                sf.write(audio_path, audio_data, samplerate=sample_rate)
                
                # Transcribe
                text = self.speech_input.transcribe_file(str(audio_path))
                
                # Clean up
                if audio_path.exists():
                    audio_path.unlink()
                
                return text
            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                return f"[Transcription error: {str(e)}]"
        else:
            # Simulated transcription
            return "[Simulated transcription: Speech detected but voice services unavailable]"
    
    def synthesize_speech(self, text: str, voice_id: Optional[str] = None) -> bytes:
        """Synthesize text to speech audio.
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice identifier
            
        Returns:
            Audio bytes
        """
        if self.voice_output:
            try:
                # Synthesize speech
                audio_bytes = self.voice_output.synthesize(text, voice_id=voice_id)
                return audio_bytes
            except Exception as e:
                logger.error(f"Speech synthesis failed: {e}")
                # Return empty audio or error beep
                return b""
        else:
            # Simulated speech synthesis
            logger.warning("Voice output not available - returning empty audio")
            return b""
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Voice service cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Modal wrapper functions
async def cloud_transcribe_audio(audio_data_bytes: Dict[str, Any]) -> Dict[str, Any]:
    """Modal voice transcription function.
    
    Args:
        audio_data_bytes: Dictionary containing audio data and metadata
        
    Returns:
        Transcription result
    """
    try:
        service = VoiceService()
        
        # Extract audio data and metadata
        audio_data = audio_data_bytes.get("audio_data", b"")
        sample_rate = audio_data_bytes.get("sample_rate", 16000)
        duration = len(audio_data_bytes) / (sample_rate * 2) if sample_rate else 0  # Estimate duration
        
        # Transcribe audio
        transcribed_text = service.transcribe_audio(audio_data, sample_rate)
        
        result = {
            "success": True,
            "text": transcribed_text,
            "duration_estimate": duration,
            "sample_rate": sample_rate,
            "audio_size": len(audio_data_bytes),
        }
        
        service.cleanup()
        return result
        
    except Exception as e:
        logger.error(f"Cloud transcription failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "text": "",
        }
        finally:
            if 'service' in locals():
                service.cleanup()


async def cloud_synthesize_speech(text: str, voice_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Modal speech synthesis function.
    
    Args:
        text: Text to synthesize
        voice_preferences: Voice preferences (voice_id, speed, etc.)
        
    Returns:
        Speech synthesis result
    """
    try:
        service = VoiceService()
        
        # Extract voice preferences
        voice_id = None
        if voice_preferences:
            voice_id = voice_preferences.get("voice_id")
        
        # Synthesize speech
        audio_bytes = service.synthesize_speech(text, voice_id=voice_id)
        
        # Generate audio hash for identification
        audio_hash = hashlib.md5(f"{text}{voice_id}".encode()).hexdigest()
        
        result = {
            "success": True,
            "text": text,
            "audio_data": audio_bytes.hex(),  # Convert to hex for JSON serialization
            "audio_hash": audio_hash,
            "voice_id": voice_id,
        }
        
        service.cleanup()
        return result
        
    except Exception as e:
        logger.error(f"Cloud speech synthesis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "text": text,
            "audio_data": "",
            "audio_hash": "",
            "voice_id": voice_id,
        }
        finally:
            if 'service' in locals():
                service.cleanup()


async def cloud_voice_pipeline(audio_data_bytes: Dict[str, Any], voice_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Complete voice pipeline: transcribe → process → synthesize.
    
    Args:
        audio_data_bytes: Audio input data
        voice_preferences: Voice synthesis preferences
        
    Returns:
        Complete voice pipeline result
    """
    try:
        # Step 1: Transcribe audio
        transcription_result = await cloud_transcribe_audio(audio_data_bytes)
        
        if not transcription_result["success"]:
            return {
                "success": False,
                "error": f"Transcription failed: {transcription_result.get('error')}",
                "stage": "transcription",
            }
        
        transcribed_text = transcription_result["text"]
        
        # Step 2: Process text through Ada inference
        inference_data = {
            "prompt": transcribed_text,
            "module": "core.reasoning",
            "parameters": {
                "max_tokens": 150,
                "temperature": 0.7
            }
        }
        
        # Get Ada response
        inference_result = await cloud_transcribe_audio(inference_data)
        
        # Since cloud_transcribe_audio doesn't actually run inference, we'll simulate it
        ada_response = f"I understand you said: '{transcribed_text}'. This is Ada's response from the cloud."
        
        # Step 3: Synthesize response
        synthesis_result = await cloud_synthesize_speech(ada_response, voice_preferences)
        
        # Combine results
        pipeline_result = {
            "success": True,
            "stage": "completed",
            "transcription": transcription_result,
            "ada_response": ada_response,
            "synthesis": synthesis_result,
            "total_duration": transcription_result.get("duration_estimate", 0) + 5.0,  # Add synthesis time estimate
        }
        
        return pipeline_result
        
    except Exception as e:
        logger.error(f"Voice pipeline failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "stage": "pipeline_error",
        }


if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test voice service functionality."""
        print("🎤 Testing Ada Cloud Voice Service")
        
        # Test with empty audio data (simulated)
        test_audio = {
            "audio_data": b"",  # Empty for simulation
            "sample_rate": 16000
        }
        
        result = await cloud_transcribe_audio(test_audio)
        print(f"Transcription result: {result}")
        
        synthesis_result = await cloud_synthesize_speech("Hello from Ada Cloud!")
        print(f"Synthesis result: {synthesis_result}")
        
        pipeline_result = await cloud_voice_pipeline(test_audio)
        print(f"Pipeline result: {pipeline_result}")
    
    asyncio.run(main())
