import speech_recognition as sr

def extract_text_from_audio(audio_file):
    recognizer = sr.Recognizer()
    
    # Load audio file
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        
    # Recognize speech using Google Speech Recognition
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Example usage
if __name__ == "__main__":
    audio_file = "03-01-01-01-01-01-01.wav"  # Update with the path to your audio file
    extracted_text = extract_text_from_audio(audio_file)
    print("Extracted text:", extracted_text)
