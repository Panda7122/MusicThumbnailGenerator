from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from bs4 import BeautifulSoup
import os
import re
import sys

def search_genius_url_selenium(artist, song_name):
    query = f"{artist} {song_name} lyrics site:genius.com"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 無頭模式
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install())) #, options=options
    driver.get(f"https://www.bing.com/search?q={query}")

    time.sleep(3)  # 等待 JS 加載結果

    links = driver.find_elements(By.CSS_SELECTOR, "a")

    genius_url = None
    for link in links:
        href = link.get_attribute("href")
        print(href)
        if href and "genius.com" in href and "lyrics" in href:
            # 有時候 Google 會加上跳轉參數，擷取純網址
            m = re.search(r"(https://genius\.com/[^&]+)", href)
            if m:
                genius_url = m.group(1)
                break

    driver.quit()
    return genius_url
def search_genius_lyrics_url(artist, song_name):
    artist = re.sub(r"[^\w\s-]", "", artist)
    song_name = re.sub(r"[^\w\s-]", "", song_name)
    artist = artist.replace(" ", "-")
    song_name = song_name.replace(" ", "-")
    return f'https://genius.com/{artist}-{song_name}-lyrics'
def get_lyrics_from_genius(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    # 嘗試找出所有歌詞段落
    lyrics_divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    # 移除包含在 LyricsHeader__Container-sc-d6abeb2b-1 fsbvCW 類別中的所有文字
    for header in soup.find_all("div", class_=re.compile(r"^LyricsHeader__Container-")):
        header.decompose()
    lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])
    lyrics = re.sub(r"\[.*?\]", "", lyrics, flags=re.DOTALL)
    lyrics = re.sub(r"\((?:.|\n)*?\)", lambda m: m.group(0).replace('\n', ' '), lyrics)
    return lyrics.strip() if lyrics else None

def save_lyrics(artist, song_name, lyrics):
    filename = "./lyrics/"+f"{artist}-{song_name}.txt".replace("/", "_")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(lyrics)
    print(f"save at {filename}")
def get_and_save_lyrics(artist, song_name):
    url = search_genius_lyrics_url(artist, song_name)
    lyrics = get_lyrics_from_genius(url)
    if lyrics:
        save_lyrics(artist, song_name, lyrics)
    else:
        
        url = search_genius_url_selenium(artist, song_name)
        if url:
            print(f"find {artist}-{artist} at {url}")
            lyrics = get_lyrics_from_genius(url)
            if lyrics:
                save_lyrics(artist, song_name, lyrics)
            else:
                return 0
        else:
            return 0
    return 1
def main():
    if len(sys.argv) < 3:
        print("Usage: python getlyrics.py <artist> <title>")
        sys.exit(1)
    artist = sys.argv[1]
    title = sys.argv[2]
    # Check if lyrics file already exists, skip if so
    filename = "./lyrics/"+f"{artist}-{title}.txt".replace("/", "_")
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping.")
        return
    if(get_and_save_lyrics(artist, title) == 0):
        print(f'error in download lyrics of {artist}-{title}')
    else:
        print(f'success download lyrics of {artist}-{title} at {filename}')
if __name__ == "__main__":
    main()
