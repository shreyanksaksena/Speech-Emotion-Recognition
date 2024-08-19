import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment, effects
import os, sys
import numpy as np
from tensorflow import keras
import tensorflow as tf
import speech_recognition as sr
from gtts import gTTS
import os

model = keras.models.load_model('speech_emotion.h5')

plt.style.use('ggplot')

audio_path = sys.argv[1]

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

def text_to_audio(text, output_file):
    tts = gTTS(text=text, lang='en')  # Language code can be changed as needed
    tts.save(output_file)

def preprocess_audio(path):
    _, sr = librosa.load(path)
    raw_audio = AudioSegment.from_file(path)
    
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    padded = np.pad(trimmed, (0, 180000-len(trimmed)), 'constant')
    return padded, sr

emotion_dic = {
    'neutral' : 0,
    'happy'   : 1,
    'sad'     : 2, 
    'angry'   : 3, 
    'fear'    : 4, 
    'disgust' : 5
}

labels = ['neutral', 'calm', 'sad', 'happy', 'fear', 'disgust']

def encode(label):
    return emotion_dic.get(label)

def main():
    zcr_list = []
    rms_list = []
    mfccs_list = []

    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    y, sr = preprocess_audio(audio_path)
    extracted_text = extract_text_from_audio(audio_path)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)

    zcr_list.append(zcr)
    rms_list.append(rms)
    mfccs_list.append(mfccs)

    X = np.concatenate((
        np.swapaxes(zcr_list, 1, 2), 
        np.swapaxes(rms_list, 1, 2), 
        np.swapaxes(mfccs_list, 1, 2)), 
        axis=2
    )
    X = X.astype('float32')

    # Evaluate the model on the test dataset
    y_pred = model.predict(X)

    # Convert predicted probabilities to class labels
    y_pred_labels = tf.argmax(y_pred, axis=1)

    output_file = "output_audio.mp3"  # Output audio file name
    text_to_audio(extracted_text, output_file)

    print(labels[y_pred_labels.numpy().tolist()[0]])

if __name__ == "__main__":
    main()