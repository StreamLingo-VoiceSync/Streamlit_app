import whisper

model = whisper.load_model("large-v3", device="cpu")

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]
