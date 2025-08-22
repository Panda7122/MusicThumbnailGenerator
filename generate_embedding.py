import torch
import pandas as pd
import os
from diffusers import StableDiffusionPipeline

def main():
    print('start running...')
    datasetPath = './kaggleData/small_song_lyrics.csv'
    dataset = pd.read_csv(datasetPath)
    # Load Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    for index, row in dataset.iterrows():
        artist = row['Artist'].replace('/', '_').replace('.', '_')
        title = row['Title'].replace('/', '_').replace('.', '_')
        input_prompt_dir = './input/prompt'
        input_prompt_path = os.path.join(input_prompt_dir, f"{artist}-{title}.txt")
        if not os.path.exists(input_prompt_path):
            continue
        with open(input_prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
        inputs = tokenizer(prompt, return_tensors="pt", max_length=77, truncation=True)
        with torch.no_grad():
            embeddings = text_encoder(**inputs).last_hidden_state
        print(f"Embedding for {artist}-{title}: {embeddings.shape}")
        output_embedding_dir = './input/trueEmbedding'
        os.makedirs(output_embedding_dir, exist_ok=True)
        output_embedding_path = os.path.join(output_embedding_dir, f"{artist}-{title}.pt")
        torch.save(embeddings.cpu(), output_embedding_path)
if __name__ == "__main__":
    main()
