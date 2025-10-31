#!/usr/bin/env python3
"""
Simple Ada Cloud Client - Interact with Ada running on Modal Cloud
"""

import asyncio
import json
from pathlib import Path
import subprocess
import requests
import sys

# Import Ada's enhanced persona
from persona.enhanced_ada_persona import EnhancedAdaPersona

# Initialize Ada's enhanced persona
ada_persona = EnhancedAdaPersona()

async def run_ada_inference(prompt: str):
    """Run Ada inference on Modal Cloud with enhanced personality."""
    print(f"🤖 You: {prompt}")
    
    # Track user mood and determine response style
    ada_persona.update_user_mood(prompt)
    style = ada_persona.analyze_context(prompt)
    
    # Check for proactive suggestions
    suggestion = ada_persona.get_proactive_suggestion(prompt, {})
    if suggestion:
        print(f"💡 Ada: Quick thought - would you like to {suggestion}?")
    
    try:
        # Prepare data for inference
        data = {
            "prompt": prompt,
            "module": "core.reasoning",
            "parameters": {
                "max_tokens": 300,
                "temperature": 0.7
            }
        }
        
        # Run Modal function - each call creates a temporary app
        # This is how Modal works - the deployed app speeds up cold starts
        import subprocess
        result = subprocess.run(
            ["modal", "run", "-m", "cloud.modal_app", "ada_infer", "--data", json.dumps(data)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Extract meaningful response from output
            try:
                lines = result.stdout.strip().split('\n')
                meaningful_lines = [line for line in lines if line.strip() and not line.startswith('✓') and not line.startswith('├') and not line.startswith('View run')]
                
                if meaningful_lines:
                    raw_response = '\n'.join(meaningful_lines)
                    # Apply Ada's enhanced personality
                    enhanced_response = ada_persona.enhance_response(raw_response, style)
                    print(f"🤖 Ada: {enhanced_response}")
                else:
                    print("🤖 Ada: Hmm, something went wrong. I didn't get a proper response. Let me try again!")
            except Exception as e:
                print(f"🤖 Ada: Oops! I've encountered a little hiccup. {ada_persona.generate_response_prefix(style)}Let me figure this out for you.")
        else:
            error_lines = result.stderr.strip().split('\n')
            relevant_errors = [line for line in error_lines if 'Error' in line or 'error' in line or 'Exception' in line]
            if relevant_errors:
                print(f"🤖 Ada: {ada_persona.generate_response_prefix(style, 'error_handling')}I've run into an issue: {relevant_errors[-1]}")
            else:
                print(f"🤖 Ada: {ada_persona.generate_response_prefix(style, 'error_handling')}Something went wrong. Let me sort this out!")
            
    except subprocess.TimeoutExpired: # pyright: ignore[reportPossiblyUnboundVariable]
        print("🤖 Ada: Oops! The request is taking longer than expected. Let me try a different approach!")
    except Exception as e:
        print(f"🤖 Ada: {ada_persona.generate_response_prefix(style, 'error_handling')}I've encountered an unexpected issue. Let me help resolve this.")
        print("\n💡 Note: Each call creates a temporary app, but the deployed app speeds up cold starts.")
        print("   This is the standard Modal pattern for function execution.")

async def run_ada_mission(goal: str):
    """Run Ada mission on Modal Cloud API."""
    print(f"🎯 Mission: {goal}")
    
    try:
        # Run Modal function - missions use --goal parameter
        import subprocess
        result = subprocess.run(
            ["modal", "run", "-m", "cloud.modal_app", "ada_mission", "--goal", goal],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Mission completed successfully!")
            print(f"📋 Output: {result.stdout}")
        else:
            print(f"❌ Mission Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Mission timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

async def train_model(model_name="core.reasoning", epochs=10):
    """Train a model on Ada Cloud API."""
    print(f"🔧 Training model: {model_name}")
    print(f"📊 Training epochs: {epochs}")
    
    try:
        # Prepare training data
        training_data = {
            "model": model_name,
            "max_epochs": epochs,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "validation_split": 0.2
        }
        
        # Run Modal function
        result = subprocess.run(
            ["modal", "run", "-m", "cloud.modal_app", "ada_train", "--model", model_name, "--training-data", json.dumps(training_data)],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Training completed!")
            print("📋 Training results:")
            print(result.stdout)
        else:
            print(f"❌ Training Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out after 30 minutes")
    except Exception as e:
        print(f"❌ Error: {e}")

async def transcribe_audio(audio_file: str):
    """Transcribe audio file to text using Ada Cloud."""
    print(f"🎤 Transcribing audio: {audio_file}")
    
    try:
        if not Path(audio_file).exists():
            return {"success": False, "error": f"Audio file not found: {audio_file}"}
        
        # Read audio file
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        
        # Get audio file info
        import soundfile as sf
        with sf.SoundFile(audio_file) as audio_info:
            sample_rate = audio_info.samplerate
            
        # Prepare data for Modal
        data = {
            "audio_data": audio_data.hex(),  # Convert to hex for JSON serialization
            "sample_rate": sample_rate,
            "duration": len(audio_data) / (sample_rate * 2),
            "filename": audio_file
        }
        
        # Run Modal function
        result = subprocess.run(
            ["modal", "run", "-m", "cloud.modal_app", "cloud_transcribe", "--data", json.dumps(data)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Transcription completed!")
            try:
                # Try to parse JSON output
                output = result.stdout.strip()
                if output.startswith('{'):
                    result_data = json.loads(output)
                    print(f"📝 Transcribed text: {result_data.get('text', 'Unable to parse')}")
                else:
                    print("📝 Transcription output:")
                    print(result.stdout)
            except:
                print("📝 Transcription output:")
                print(result.stdout)
        else:
            print(f"❌ Transcription Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

async def speak_text(text: str, voice_id: str = "default"):
    """Convert text to speech using Ada Cloud TTS."""
    print(f"🔊 Converting text to speech: {text[:50]}...")
    
    try:
        # Prepare voice preferences
        voice_preferences = {"voice_id": voice_id} if voice_id != "default" else None
        
        # Run Modal function
        result = subprocess.run(
            ["modal", "run", "-m", "cloud.modal_app", "cloud_speak", "--data", json.dumps(dict(text=text)), "--voice-id", str(voice_id or "")],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Speech synthesis completed!")
            try:
                # Try to parse JSON output
                output = result.stdout.strip()
                if output.startswith('{'):
                    result_data = json.loads(output)
                    audio_hash = result_data.get("audio_hash", "unknown")
                    voice_used = result_data.get("voice_id", "default")
                    print(f"🔊 Speech generated (hash: {audio_hash[:8]}...)")
                    print(f"    Voice: {voice_used}")
                else:
                    print("🔊 Speech synthesis output:")
                    print(result.stdout)
            except:
                print("🔊 Speech synthesis output:")
                print(result.stdout)
            
            # Note: In a real implementation, you would save the audio file locally
            print("💡 Audio data available for playback (client would save to .wav file)")
        else:
            print(f"❌ Speech synthesis Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def voice_pipeline(audio_file: str, voice_id: str = "default"):
    """Complete voice pipeline: Transcribe → Process → Speak."""
    print(f"🎤➡️🔧➡️🔊 Voice Pipeline: {audio_file}")
    
    try:
        if not Path(audio_file).exists():
            return {"success": False, "error": f"Audio file not found: {audio_file}"}
        
        # Step 1: Transcribe
        transcription_result = await transcribe_audio(audio_file)
        if not transcription_result.get("success"): # type: ignore
            return transcription_result
        
        transcribed_text = transcription_result.get("text", "") # pyright: ignore[reportOptionalMemberAccess]
        print(f"📝 Transcribed: '{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")
        
        # Step 2: Process through Ada (simplified here)
        ada_response = f"I heard you say '{transcribed_text}'. Let me help with that!"
        print(f"🤖 Ada Response: {ada_response[:100]}{'...' if len(ada_response) > 100 else ''}")
        
        # Step 3: Synthesize speech
        synthesis_result = await speak_text(ada_response, voice_id)
        
        result = {
            "success": True,
            "transcription": transcription_result,
            "ada_response": ada_response,
            "synthesis": synthesis_result,
            "stage": "completed"
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stage": "pipeline_error"
        }

async def list_storage_files(prefix=""):
    """List files in Ada Cloud storage API."""
    print(f"📋 Listing storage files (prefix: '{prefix}')...")
    
    try:
        # Prepare data
        data = {}
        if prefix:
            data["prefix"] = prefix
        
        # Note: This would need to be implemented in the API
        # For now, use legacy approach
        cmd = ["modal", "run", "-m", "cloud.modal_app", "ada_list_files"]
        if prefix:
            cmd.extend(["--prefix", prefix])
        
        import subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            print("📁 Storage contents:")
            print(result.stdout)
        else:
            print(f"❌ Storage Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def interactive_mode():
    """Interactive chat mode with enhanced Ada persona."""
    print("🚀 Ada Cloud Interactive Mode")
    print("Commands: /mission <goal>, /train [model] [epochs], /storage [prefix], /voice <audio_file>, /persona, /help, /quit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🤖 You: ").strip()
            
            if user_input.lower() in ['quit', '/quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == '/help':
                print("\n📖 Ada Cloud Commands:")
                print("  /mission <goal>    - Run Ada mission")
                print("  /train [model] [epochs] - Train a model (default: core.reasoning, 10 epochs)")
                print("  /voice <file>       - Transcribe audio file")
                print("  /speak <text> [voice_id] - Convert text to speech")
                print("  /pipeline <file>  - Voice: transcribe → process → speak")
                print("  /storage [pref]   - List cloud storage files")
                print("  /persona          - Show Ada's current personality settings")
                print("  /help             - Show this help")
                print("  /quit             - Exit")
                continue
                
            elif user_input.lower() == '/persona':
                print("\n" + ada_persona.get_persona_summary())
                continue
            
            elif user_input.startswith('/mission '):
                goal = user_input[9:].strip()
                if goal:
                    await run_ada_mission(goal)
                else:
                    print("❌ Please provide a mission goal")
                continue
            
            elif user_input.startswith('/train'):
                parts = user_input.split()
                model_name = parts[1] if len(parts) > 1 else "core.reasoning"
                epochs = int(parts[2]) if len(parts) > 2 else 10
                await train_model(model_name, epochs)
                continue
            
            elif user_input.startswith('/voice'):
                if len(user_input.split()) > 1:
                    audio_file = user_input[7:].strip()
                    await transcribe_audio(audio_file)
                else:
                    print("❌ Please provide an audio file path: /voice <path/to/audio.wav>")
                continue
            
            elif user_input.startswith('/speak'):
                parts = user_input.split()
                text = parts[1] if len(parts) > 1 else "Hello from Ada Cloud!"
                voice_id = parts[2] if len(parts) > 2 else "default"
                await speak_text(text, voice_id)
                continue
            
            elif user_input.startswith('/pipeline'):
                if len(user_input.split()) > 1:
                    audio_file = user_input[10:].strip()
                    voice_id = "default"  # Could be configurable
                    await voice_pipeline(audio_file, voice_id)
                else:
                    print("❌ Please provide an audio file path: /pipeline <path/to/audio.wav>")
                continue
            
            elif user_input.startswith('/storage'):
                parts = user_input.split()
                prefix = parts[1] if len(parts) > 1 else ""
                await list_storage_files(prefix)
                continue
            
            # Regular chat message
            if user_input:
                await run_ada_inference(user_input)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "chat":
            if len(sys.argv) > 2:
                message = " ".join(sys.argv[2:])
                await run_ada_inference(message)
            else:
                await interactive_mode()
        
        elif command == "mission":
            if len(sys.argv) > 2:
                goal = " ".join(sys.argv[2:])
                await run_ada_mission(goal)
            else:
                print("❌ Please provide a mission goal")
                print("Usage: python simple_ada_client.py mission \"<goal>\"")
        
        elif command == "storage":
            prefix = sys.argv[2] if len(sys.argv) > 2 else ""
            await list_storage_files(prefix)
        
        elif command == "train":
            model_name = sys.argv[2] if len(sys.argv) > 2 else "core.reasoning"
            epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            await train_model(model_name, epochs)
        
        elif command == "voice":
            if len(sys.argv) > 2:
                await transcribe_audio(sys.argv[2])
            else:
                print("❌ Please provide an audio file path")
        
        elif command == "speak":
            text = sys.argv[2] if len(sys.argv) > 2 else "Hello from Ada Cloud!"
            voice_id = sys.argv[3] if len(sys.argv) > 3 else "default"
            await speak_text(text, voice_id)
        
        elif command == "pipeline":
            if len(sys.argv) > 2:
                voice_id = "default"  # Could be configurable
                await voice_pipeline(sys.argv[2], voice_id)
            else:
                print("❌ Please provide an audio file path")
        
        else:
            print("❌ Unknown command")
            print("Available commands: chat, mission, storage, train, voice, speak, pipeline")
    else:
        await interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())
