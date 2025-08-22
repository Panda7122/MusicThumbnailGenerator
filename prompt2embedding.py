import torch
import transformers
import requests
import pandas as pd
import os
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print('start runing...')
    datasetPath = './kaggleData/small_song_lyrics.csv'
    # Read the dataset as a CSV file
    dataset = pd.read_csv(datasetPath)
    for index, row in dataset.iterrows():
        artist = row['Artist']
        title = row['Title']
        title = title.replace('/', '_')
        title = title.replace('.', '_')
        artist = artist.replace('/', '_')
        artist = artist.replace('.', '_')
        text_save_dir = './input/prompt'
        text_save_path = os.path.join(text_save_dir, f"{artist}-{title}.txt")
        
        with open(text_save_path, 'r', encoding='utf-8') as f:
            text = f.read()
        with autocast("cuda"):
            image = pipe(text)["sample"][0]  
        image.save(f'./output/llm/{artist}-{title}.png')