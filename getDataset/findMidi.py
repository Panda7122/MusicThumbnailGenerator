import pandas as pd
import requests
from bs4 import BeautifulSoup

csv_path = '/home/Panda/Desktop/courses/project/kaggleData/small_song_lyrics.csv'

df = pd.read_csv(csv_path)
err = []
correct = []
for index, row in df.iterrows():
    artist = row['Artist']
    title = row['Title']
    print(f"Artist: {artist}, Title: {title}")
    artist = artist.lower()
    title = title.lower()
    artist = artist.replace(' ', '-')
    title = title.replace('?', '')
    title = title.replace(' ', '-')
    # https://www.mididb.com/justin-bieber/love-yourself-midi/
    # search_url = f"https://api.mididb.com/search?artist={artist}&title={title}"
    search_url = f"https://www.mididb.com/{artist}/{title}-midi/"
    search_url2 = f"https://freemidi.org/search?q={artist}+{title}"
    print(search_url)
    response = requests.get(search_url)

    if response.status_code == 200:
        Soup = BeautifulSoup(response.text,'html.parser') 
        # print(Soup)
        midi_links = Soup.find_all('a', href=True)
        for link in midi_links:
            if 'midi' in link['href']:
                midi_url = link['href']
                if midi_url.endswith('.mid'):
                    with open(f"../trainingData/midi/{title}.mid", 'wb') as midi_file:
                        midi_file.write(requests.get(midi_url).content)
                    print(f"MIDI URL: {midi_url}")
    else:
        print("Error fetching MIDI data.")
        nowerr = {'artist': artist, 'title': title}
        err.append(nowerr)
err_df = pd.DataFrame(err)
err_df.to_csv('./errlist.csv', index=False)