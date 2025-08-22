import pretty_midi
import librosa
import numpy as np
import pandas as pd
import os
from audio_to_midi import AudioToMidi

def audio_to_midi(title,artist):
    file_in = f'./trainingData/audio/{artist}-{title}.wav'
    file_out = f'./trainingData/midi/{artist}-{title}.mid'

    # Load the audio file
    y, sr = librosa.load(file_in)

    # Estimate the tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Get the duration of the audio
    duration = librosa.get_duration(y=y, sr=sr)

    converter = AudioToMidi()
    converter.convert(file_in, file_out)
csv_path = '/home/Panda/Desktop/courses/project/kaggleData/small_song_lyrics.csv'
df = pd.read_csv(csv_path)
for index, row in df.iterrows():
    artist = row['Artist']
    title = row['Title']
    print(f"Artist: {artist}, Title: {title}")
    wav_file_path = f'./trainingData/audio/{artist}-{title}.wav'
    midi_file_path = f'./trainingData/midi/{artist}-{title}.mid'
    if os.path.exists(wav_file_path):
        if not os.path.exists(midi_file_path):
            print(f"{midi_file_path} not exists, make midifile.")
            audio_to_midi(title, artist)
