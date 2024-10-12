import pandas as pd
import requests
import re
import urllib.request
import urllib.parse
import os
import ssl
from bs4 import BeautifulSoup
from youtubesearchpython.__future__ import Search
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from pytubefix import YouTube
from pytubefix.cli import on_progress

import subprocess 
import tensorflow as tf
# from basic_pitch.inference import predict_and_save,  Model
# from basic_pitch import ICASSP_2022_MODEL_PATH
# basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

import librosa
ssl._create_default_https_context = ssl._create_stdlib_context
def mp3toWav(title, artist):
    audio_dir = "../trainingData/audio/"
    mp3File = None
    for file in os.listdir(audio_dir):
        if file.startswith(artist + '-' + title) and file.endswith('.mp3'):
            mp3File = "../trainingData/audio/"+artist+'-'+title +'.mp3'
            break
    if mp3File == None:
        mp3File = "../trainingData/audio/"+artist+'-'+title +'.mp4'
        
    if not mp3File:
        raise FileNotFoundError(f"No mp4 file found for {artist} - {title}")
    wavFile = "../trainingData/audio/"+artist+'-'+title + '.wav'
    midFile = "../trainingData/midi/"
    
    subprocess.call(['ffmpeg', '-i', mp3File, 
                 wavFile])
    if os.path.exists(mp3File):
        os.remove(mp3File)
        print(f"Removed {mp3File}")
    else:
        print(f"{mp3File} does not exist")
    # print('generating midi')
    # y, sr = librosa.load(wavFile)

    # # Estimate the tempo (BPM)
    # tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # print(f'tempo: {tempo}')
    # # Get the duration of the audio
    # predict_and_save([wavFile], midFile, save_midi=True, sonify_midi=False,midi_tempo=tempo, save_model_outputs=False, save_notes=False, model_or_model_path=basic_pitch_model)
    # print('success')
    # if os.path.exists(wavFile):
    #     os.remove(wavFile)
    # # Rename the generated MIDI file
    # old_midi_file = f'../trainingData/midi/{artist}-{title}_basic_pitch.mid'
    # # trashwav = f'../trainingData/midi/{artist}-{title}_basic_pitch.wav'
    # # os.remove(trashwav)
    # # trashcsv = f'../trainingData/midi/{artist}-{title}_basic_pitch.csv'
    # # os.remove(trashcsv)
    
    # new_midi_file = f'../trainingData/midi/{artist}-{title}.mid'
    # if os.path.exists(old_midi_file):
    #     os.rename(old_midi_file, new_midi_file)
    #     print(f"Renamed {old_midi_file} to {new_midi_file}")
    # else:
    #     print(f"{old_midi_file} does not exist")
import pretty_midi
import librosa
import numpy as np
from typing import Union
# from basic_pitch.inference import Model
import pathlib

def findURL(query):
    searching = Search(query, limit = 20)
    return searching.result()
def download_audio(artist,title, url):
    
    yt = YouTube(url, on_progress_callback = on_progress)
    print('download...')
    try:
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=f'../trainingData/audio/')
        base, ext = os.path.splitext(out_file)
        print(out_file)
        new_file = '../trainingData/audio/'+artist+'-'+title + '.mp4'
        try:
            os.rename(out_file, new_file)
            print("target path = " + (new_file))
            print("mp4 has been successfully downloaded.")
        except Exception as e:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
csv_path = '../kaggleData/small_song_lyrics.csv'
df = pd.read_csv(csv_path)
err = []
correct = []
for index, row in df.iterrows():
    artist = row['Artist']
    title = row['Title']
    print(f"Artist: {artist}, Title: {title}")
    wav_file_path = f'../trainingData/audio/{artist}-{title}.wav'
    midi_file_path = f'../trainingData/midi/{artist}-{title}.mid'
    if os.path.exists(wav_file_path):
        print(f"{wav_file_path} already exists, skipping download.")
        continue
    query = f"{artist} {title} audio"
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    attempts = 0
    video_url = None
    while attempts < 2:
        try:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            driver.get("https://www.youtube.com")
            search_box = driver.find_element(By.NAME, "search_query")
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            time.sleep(2)
            video = driver.find_element(By.ID, "video-title")
            video_url = video.get_attribute("href")
            time.sleep(2)
            driver.quit()
            if video_url:
                break
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            # driver.quit()
        attempts += 1
    if video_url:
        video_url = re.sub(r'&pp=.*', '', video_url)
        print(f'find video URL {video_url}')
        correct.append((artist, title, video_url))
        print(artist, title, video_url)
        # try:
        download_audio(artist,title, video_url)
        print(f"Downloaded audio for {title} by {artist}")
        time.sleep(1)
        mp3toWav(title, artist)
        # except Exception as e:
        #     print(f'download {artist}-{title} with {video_url} fail')
        #     print(f'fail reason {e}')
        #     err.append((artist, title))
        # audioToMIDI(artist,title)
    else:
        print('not found')
        err.append((artist, title))
    # Close the driver

