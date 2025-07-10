
from gtts import gTTS
import os

def synthesize_speech(text, voice_lang='en'):
    output_file = os.path.join("audio", "output.mp3")
    tts = gTTS(text=text, lang=voice_lang)
    tts.save(output_file)
    return output_file

