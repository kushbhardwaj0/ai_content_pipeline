# 03_text_to_speech.py
from gtts import gTTS
from pydub import AudioSegment
import os

def text_to_speech(text, output_dir="output"):
    """
    Converts text to speech, speeds it up, and saves it as an MP3 file.
    """
    raw_audio_path = os.path.join(output_dir, "input_audio.mp3")
    final_audio_path = os.path.join(output_dir, "output.mp3")

    # Convert text to speech
    tts = gTTS(text=text, lang='en')
    tts.save(raw_audio_path)
    print(f"Raw audio saved to {raw_audio_path}")

    # Speed up the audio
    sound = AudioSegment.from_mp3(raw_audio_path)
    speed_up_sound = sound.speedup(playback_speed=1.3)
    speed_up_sound.export(final_audio_path, format="mp3")
    print(f"Sped-up audio saved to {final_audio_path}")

    return final_audio_path

if __name__ == "__main__":
    # Read the generated text from the previous step
    with open("output/generated_text.txt", "r") as f:
        generated_text = f.read()
    
    text_to_speech(generated_text)
