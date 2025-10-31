"""Cloud wrapper functions for voice services - bridges the cloud voice module and RPC."""

import json
from typing import Dict, Any, Optional

async def cloud_transcribe_audio(audio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Cloud wrapper for audio transcription.
    
    Args:
        audio_data: Dictionary containing audio data and metadata
        
    Returns:
        Transcription result
    """
    from cloud.voice_service import VoiceService
    
    service = VoiceService()
    return await service.transcribe_audio(audio_data)

async def cloud_synthesize_speech(text: str, voice_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cloud wrapper for speech synthesis.
    
    Args:
        text: Text to convert to speech
        voice_preferences: Voice preferences (voice_id, speed, etc.)
        
    Returns:
        Speech synthesis result
    """
    from cloud.voice_service import VoiceService
    
    service = VoiceService()
    return await service.synthesize_speech(text, voice_preferences)

async def cloud_voice_pipeline(audio_data: Dict[str, Any], voice_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cloud wrapper for complete voice pipeline.
    
    Args:
        audio_data: Audio data dictionary with audio bytes and metadata
        voice_preferences: Voice synthesis preferences
        
    Returns:
        Complete voice pipeline result
    """
    from cloud.voice_service import VoiceService
    
    service = VoiceService()
    return await service.voice_pipeline(audio_data, voice_preferences)
