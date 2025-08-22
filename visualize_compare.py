import torch
import pandas as pd
import os
def main():
    print('start running...')
    datasetPath = './kaggleData/small_song_lyrics.csv'
    dataset = pd.read_csv(datasetPath)
    # Load Stable Diffusion pipeline


    for index, row in dataset.iterrows():
        artist = row['Artist'].replace('/', '_').replace('.', '_')
        title = row['Title'].replace('/', '_').replace('.', '_')
        
if __name__ == "__main__":
    main()
