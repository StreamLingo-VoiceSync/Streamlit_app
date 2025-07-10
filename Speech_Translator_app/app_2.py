import streamlit as st
from stt import transcribe_audio
from demo_app.mt import translate_text
from gtts_module import synthesize_speech
import os
from resemblyzer import VoiceEncoder, preprocess_wav
from pydub import AudioSegment

st.title("Speech Translation App 🎙️🌍")

# Create 'audio' folder if it doesn't exist
os.makedirs("audio", exist_ok=True)

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file to disk
    audio_path = os.path.join("audio", uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(audio_path, format="audio/wav")

    # Convert MP3 to WAV if needed (for embedding extraction)
    if audio_path.endswith(".mp3"):
        wav_path = audio_path.replace(".mp3", ".wav")
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
    else:
        wav_path = audio_path

    # Step 1: STT
    st.markdown("### 1. 📝 Transcription")
    transcribed_text = transcribe_audio(audio_path)
    st.text_area("Transcribed Text", transcribed_text, height=100)

    if transcribed_text:
        # Step 2: MT
        st.markdown("### 2. 🌍 Translation")
        target_lang = st.selectbox("Select target language", ["fr", "de", "hi", "es", "zh", "en"])
        translated_text = translate_text(transcribed_text, target_lang)
        st.text_area("Translated Text", translated_text, height=100)

        if not translated_text.startswith("[ERROR]"):
            # Step 3: TTS
            st.markdown("### 3. 🗣️ Synthesized Speech")
            output_audio_path = synthesize_speech(translated_text, voice_lang=target_lang)
            st.audio(output_audio_path, format="audio/mp3")
        else:
            st.error(translated_text)

    # Step 4: Voice Embedding
    st.markdown("### 4. 🔊 Voice Embedding (193-dim vector)")
    try:
        wav = preprocess_wav(wav_path)
        encoder = VoiceEncoder()
        embedding = encoder.embed_utterance(wav)

        st.success("Voice embedding extracted successfully.")
        st.write(embedding)  # raw vector
        st.line_chart(embedding)  # plot
    except Exception as e:
        st.error(f"Embedding extraction failed: {e}")
