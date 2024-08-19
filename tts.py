from gtts import gTTS
import os

def text_to_audio(text, output_file):
    tts = gTTS(text=text, lang='en')  # Language code can be changed as needed
    tts.save(output_file)

# Example usage
if __name__ == "__main__":
    text = "Hello, how are you today?"
    output_file = "output_audio.mp3"  # Output audio file name
    text_to_audio(text, output_file)
    print("Text converted to audio and saved as", output_file)
