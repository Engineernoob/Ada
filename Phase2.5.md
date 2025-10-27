Extend Ada into a voice-interactive, reinforcement-learning assistant that can learn from speech, tone, and emotional feedback in real time.

Goal: Integrate speech recognition, voice synthesis, and sentiment-based reward signals so Ada can be trained naturally through spoken conversation.

---

### 🧠 Objectives

1. Add real-time speech-to-text input using Whisper.cpp or faster-whisper
2. Add text-to-speech output using Piper or Coqui
3. Implement voice activity detection (Silero VAD)
4. Extract sentiment / tone features from user audio and use them as RL rewards
5. Let Ada learn incrementally from spoken approval or frustration (tone-driven reward)
6. Allow toggling between Text-Only and Voice-RL mode

---

### 🧩 System Extensions

#### `/interfaces/speech_input.py`

- Capture mic input via `sounddevice` or `pyaudio`
- Run audio through Whisper.cpp → text
- Pipe text into Ada’s main event loop
- Detect start/stop speaking with Silero VAD
- Example usage:

  ```bash
  python -m ada.interfaces.speech_input --live
  /interfaces/voice_output.py

  Convert Ada’s generated text to speech via Piper/Coqui

  Cache synthesized voice samples in /storage/voice_cache/

  Add persona field in config/settings.yaml for voice tone (e.g. “warm”, “calm”)

  /rl/reward_from_voice.py

  Analyze incoming audio to infer emotional tone

  Use open-source sentiment classifier or pitch/energy metrics

  Reward example:

  if tone == "positive": reward = +1.0
  elif tone == "neutral": reward = 0.0
  else: reward = -1.0


  Return reward to RewardEngine for incremental update

  /core/event_loop.py

  Extend the loop:

  Listen → transcribe → respond → speak

  Compute reward from tone → update AdaCore weights

  Add session state for mode switching:

  mode = "voice_rl"  # or "text"

  🧩 Config Additions

  /config/settings.yaml

  voice:
    stt_backend: whispercpp
    tts_backend: piper
    vad: silero
    persona_voice: female_warm
  reinforcement:
    voice_reward: true
    reward_scale: 0.5

  ⚙️ Implementation Notes for Droid

  Use torch.mps backend when available

  Keep audio buffers short (2-3 s) for responsiveness

  Include small CLI flag:

  make run-voice


  Add lightweight visualizer in terminal showing:

  🎤 Listening...   🧠 Processing...   🎚 Reward +0.7 (Tone: Positive)


  Maintain compatibility with macOS CoreAudio drivers

  Ensure offline capability (no API calls)

  📦 Deliverables

  Speech input/output modules integrated with RL loop

  RewardFromVoice component for tone-based learning

  Updated event loop combining speech + RL inference

  Configurable persona/voice options

  README section: “Running Ada in Voice-Learning Mode”

  🧩 Phase 3 Preview

  Multi-modal perception: camera input for facial emotion → richer reward model

  Self-reflection mode: Ada critiques her tone after sessions (“Did I sound too flat?”)

  Context persistence via embeddings of voice sessions
  ```
