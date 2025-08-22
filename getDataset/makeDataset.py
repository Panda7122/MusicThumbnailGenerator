import os
import pandas as pd

# import kaggle

# kaggle.api.dataset_download_files('carlosgdcj/genius-song-lyrics-with-language-information', 
#                                   path='./', 
#                                   unzip=True)

# kaggle.api.dataset_download_files('deepshah16/song-lyrics-dataset', 
#                                   path='./', 
#                                   unzip=True)

csv_folder = '../kaggleData/csv/'
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

dataframes = []

for file in csv_files:
    file_path = os.path.join(csv_folder, file)
    print(f'reading {file_path}')
    df = pd.read_csv(file_path)
    print("filting artist, title and lyrics")
    df = df[['Artist', 'Title', 'Lyric']]
    print("appending")
    dataframes.append(df)
    # Save the combined dataframe to a CSV file
# Example: Concatenate all dataframes into a single dataframe
smalldf = pd.concat(dataframes, ignore_index=True)
smalldf.to_csv('../kaggleData/small_song_lyrics.csv', index=False)
print('reading ../kaggleData/song_lyrics.csv')
df = pd.read_csv('./kaggleData/song_lyrics.csv')
df.rename(columns={'artist': 'Artist'}, inplace=True)
df.rename(columns={'title': 'Title'}, inplace=True)
df.rename(columns={'lyrics': 'Lyric'}, inplace=True)
print("filting artist, title and lyrics")
df = df[['Artist', 'Title', 'Lyric']]
print("appending")
dataframes.append(df)
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv('../kaggleData/combined_song_lyrics.csv', index=False)
