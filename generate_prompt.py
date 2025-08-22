import torch
import transformers
import requests
import pandas as pd
import os
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

llm_name = "qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    trust_remote_code=True,
    device_map="auto"
).eval()
# if tokenizer.pad_token is None:
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eod_id
tokenizer.pad_token = '<|endoftext|>'
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))

def main():
    print('start runing...')
    datasetPath = './kaggleData/small_song_lyrics.csv'
    # Read the dataset as a CSV file
    dataset = pd.read_csv(datasetPath)
    for index, row in dataset.iterrows():
        try:
            artist = row['Artist']
            title = row['Title']
            title = title.replace('/', '_')
            title = title.replace('.', '_')
            artist = artist.replace('/', '_')
            artist = artist.replace('.', '_')
            lyric = row['Lyric']
            if os.path.exists(f'./input/prompt/{artist}-{title}.txt'):
                print(f"Skipping {artist}-{title}, file already exists.")
                continue
            prompt = f"""
I'd like you to process the following lyrics in two steps:
1. Analyze the lyrics for themes, emotions, settings, and symbolic imagery.
2. Based on the analysis, write an English prompt suitable for Stable Diffusion that describes a visual scene inspired by the lyrics.
Lyrics:
{lyric}
response prompt only
"""
            print(f'input:{prompt}')
            torch.cuda.empty_cache()
            
            response, history = model.chat(tokenizer, prompt, history=None)

            if response[:13].lower() == 'visual scene:' or response[:12].lower() == 'visual scene':
                response = response[13:] if response[:13].lower() == 'visual scene:' else response[12:]
            if response[:7].lower() == 'prompt:' or response[:6].lower() == 'prompt':
                response = response[7:] if response[:7].lower() == 'prompt:' else response[6:]
            print("content:\n", response)
            text_save_dir = './input/prompt'
            os.makedirs(text_save_dir, exist_ok=True)
            text_save_path = os.path.join(text_save_dir, f"{artist}-{title}.txt")
            with open(text_save_path, 'w+') as f:
                f.write(response)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing {artist}-{title}: {e}")
            error_save_dir = './error/get_true_space_7B'
            os.makedirs(error_save_dir, exist_ok=True)
            error_save_path = os.path.join(error_save_dir, f"{artist}-{title}.txt")
            with open(error_save_path, 'w+') as ef:
                ef.write(str(e))
            traceback.print_exc()
        # torch.save(response, text_save_path)
if __name__ == "__main__":
    main()