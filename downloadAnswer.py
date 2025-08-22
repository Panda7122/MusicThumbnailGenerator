import time
from pytubefix import YouTube
from pytubefix.cli import on_progress
import pandas as pd
from youtubesearchpython.__future__ import Search
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import re
import requests
import os
def findURL(query):
    searching = Search(query, limit = 20)
    return searching.result()

def downloadThumbnail(artist,title, url):
    yt = YouTube(url, on_progress_callback = on_progress, use_po_token=True)
    thumbnail_url = yt.thumbnail_url
    response = requests.get(thumbnail_url)
    if response.status_code == 200:
        filename = f"./trainingData/thumbnails/{artist}-{title}.jpg"
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Thumbnail saved to {filename}")
    else:
        print("Failed to download thumbnail.")
csv_path = './kaggleData/small_song_lyrics.csv'
df = pd.read_csv(csv_path)
for index, row in df.iterrows():
    artist = row['Artist']
    title = row['Title']
    title = title.replace('/', '_')
    # title = title.replace('’', '')
    title = title.replace('.', '_')
    # title = title.replace('', '_')
    # title = title.replace(' _', '_')
    # title = title.replace('_ ', '_')
    # title = title.replace(' ', '_')
    print(f"Artist: {artist}, Title: {title}")
    query = f"{artist} {title}"
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
        downloadThumbnail(artist,title, video_url)