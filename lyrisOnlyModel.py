import torch
import torch.nn as nn
import clip
from torch.nn.functional import cosine_similarity
import torch.optim as optim

from transformers import BertTokenizer, BertModel
from diffusers import StableDiffusionPipeline, DDPMScheduler
import pretty_midi
import sys
import os
from datetime import datetime
import pandas as pd
device = "cuda"
cacheLoc = ".cache"
class MusicBERT2DiffusionAdapterWithCLIP(nn.Module):
    

    def __init__(self, hidden_dim=1024, embedding_dim=768, bert_dim = 768,clip_model_name='ViT-B/32'):
        super(MusicBERT2DiffusionAdapterWithCLIP, self).__init__()
        # latent_dim of picture is 4*64*64
        # MusicBERT to Diffusion linear layer
        latent_dim=16384
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bert_dim = bert_dim
        self.latent_dim = latent_dim
        
        # embedding layer
        self.linear1 = nn.Linear(bert_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(embedding_dim, hidden_dim)
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=4, stride=2, padding=1) # conv 4*16*16=>4*32*32
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=4, stride=2, padding=1)  # conv 32*32*4 => 64*64*4
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device, download_root=cacheLoc)
        self.clip_model.eval()
    def _padding(self, token_vectors):
        batch_size, seq_len, hidden_size = token_vectors.size()
        if seq_len < self.token_dim:
            padding_size = self.max_length - seq_len
            padding = torch.zeros(batch_size, padding_size, hidden_size, device=token_vectors.device)
            token_vectors = torch.cat((token_vectors, padding), dim=1)
        elif seq_len > self.token_dim:
            token_vectors = token_vectors[:, :self.token_dim, :]
        return token_vectors

    def forward(self,  lyris_vector):
        lyris_vector = lyris_vector.to(torch.float32)
        tokenSeq = self.linear1(lyris_vector) # 768 to 768
        tokenSeq = self.relu(tokenSeq) 
        
        linear_vector = self.linear2(tokenSeq) # 768 to 1024
        linear_vector = self.relu(linear_vector)
        
        latent_vector = linear_vector.view(-1, 4, 16, 16) #reshape to 4*16*16
        latent_vector = self.conv_transpose1(latent_vector) # convolution to 4*32*32
        latent_vector = self.relu(latent_vector)
        
        latent_vector = self.conv_transpose2(latent_vector) # convolution to 4*64*64
        latent_vector = self.relu(latent_vector)
        return (latent_vector,tokenSeq)# diffusor_latent,  token seq
    def get_diffusion_input(self, lyris_vector):
        diffusion_input,token = self.forward(lyris_vector)
        return diffusion_input, token
    
    def calculate_similarity_loss(self, lyris_vector, images):
        images = self.clip_preprocess(images)
        images = images.to(device)  # Move images to the same device as the model
        image_clip_embeds = self.clip_model.encode_image(images)  # [batch_size, clip_embed_dim]
        text = self.clip_model.encode_text(lyris_vector)
        image_clip_embeds = image_clip_embeds / image_clip_embeds.norm(dim=-1, keepdim=True)
        text_clip_embeds = text / text.norm(dim=-1, keepdim=True)

        similarity = (text_clip_embeds @ image_clip_embeds.T) 
        # print(similarity)
        # similarity = self.clip_model(text, image_clip_embeds)
        # similarity = cosine_similarity(text, image_clip_embeds, dim=-1)
        one = torch.ones_like(similarity)
        similarity_loss = torch.sub(one, similarity).mean()

        return similarity_loss
    def load_model(model_path):
        model = torch.load(model_path)
        model.eval()  # 設置為評估模式
        return model
    def save_dict_model(self, model_path):
        torch.save(self.state_dict(), model_path)
    def save_Model(self, model_path):
        torch.save(self, model_path)


def traning(lyrisfile:str, BERT, bertTokenizer, Adaptermodel:MusicBERT2DiffusionAdapterWithCLIP, diffusionModel, songName, artist, learningRate):
    # Initialize the model
    # note_sequence = midi_to_note_sequence(midifile)
    # tokenization the note
    print(f'start training with {artist}-{songName}')
    lyrisinputs = bertTokenizer(lyrisfile, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        lyrisinputs = {key: value.to(device) for key, value in lyrisinputs.items()}
        lyris_vector = BERT(**lyrisinputs).last_hidden_state[:, 0, :].to(device, dtype=torch.float32)
    # Get diffusion input
    print('done Tokenizer')
    diffusion_input, TOKEN = Adaptermodel.get_diffusion_input(lyris_vector)
    print("Diffusion Input Shape:", diffusion_input.shape)
    print('done forward')
    # diffusion_input = diffusion_input.view(diffusion_input.size(0), -1) 
    # Generate images using the diffusion model
    stringOfPrompt = f'cover for "{songName}" and artist is "{artist}", show "{songName}" as title and {artist} as subtitle'
    
    diffusion_input = diffusion_input.to(device, dtype=diffusionModel.unet.dtype)
    images = diffusionModel(prompt=stringOfPrompt, latents=diffusion_input).images
    print('done images')
    
    # Calculate similarity loss

    similarity_loss = Adaptermodel.calculate_similarity_loss(TOKEN, images[0])
    image_clip_embeds = Adaptermodel.clip_model.encode_image(images[0])  # [batch_size, clip_embed_dim]
    optimizer = optim.Adam([TOKEN, image_clip_embeds], lr=learningRate)
    optimizer.zero_grad()
    similarity_loss.backward() # calculate gradient
    optimizer.step()

    print('done backward')
    print("Similarity Loss:", similarity_loss.item())
    print('saving model')
    nowtime = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./savePoint/model_{nowtime}.pt"
    Adaptermodel.save_dict_model(model_path)
    # Save the generated images
    for i, image in enumerate(images):
        image_path = f"./generated_images/{songName}_{artist}_{i}.png"
        image.save(image_path)
        print(f"Image saved at {image_path}")
def predict(lyrisfile:str, BERT, bertTokenizer, Adaptermodel:MusicBERT2DiffusionAdapterWithCLIP, diffusionModel, songName, artist):
    # Initialize the model
    lyrisinputs = bertTokenizer(lyrisfile, return_tensors="pt", padding=True, truncation=True, max_length=Adaptermodel.bert_dim)
    
    with torch.no_grad():
        lyris_vector = BERT(**lyrisinputs).last_hidden_state[:, 0, :].to(device)
    # Get diffusion input
    diffusion_input = Adaptermodel.get_diffusion_input(lyris_vector)
    print("Diffusion Input Shape:", diffusion_input.shape)
    diffusion_input = diffusion_input.view(diffusion_input.size(0), -1) 
    # Generate images using the diffusion model
    stringOfPromot = f'cover for "{songName}" and artist is "{artist}", show "{songName}" as title and {artist} as subtitle'
    
    images = diffusionModel(prompt=stringOfPromot, latents=diffusion_input).images
    # Calculate similarity loss
    similarity_loss = Adaptermodel.calculate_similarity_loss(lyris_vector, images)
    print("Similarity Loss:", similarity_loss.item())
    return images, similarity_loss.item()
def main():
    print(f'start with device {device}')
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased",cache_dir=cacheLoc)
    bert_model = BertModel.from_pretrained("google-bert/bert-base-multilingual-uncased",cache_dir=cacheLoc)
    model = MusicBERT2DiffusionAdapterWithCLIP()
    diffusionModelName = "CompVis/stable-diffusion-v1-4"
    diffusionModel = StableDiffusionPipeline.from_pretrained(diffusionModelName,
                                               variant="fp16", torch_dtype=torch.float16, cache_dir=cacheLoc)
    print('done init')
    
    # Move models to the appropriate device
    bert_model.to(device)
    model.to(device)
    diffusionModel.to(device)
    bert_model.eval()
    # Directory containing MIDI files
    lyrics_df = pd.read_csv('./kaggleData/small_song_lyrics.csv')

    # Iterate over the lyrics
    for index, row in lyrics_df.iterrows():
        # Artist,Title,Lyric
        lyrisfile = row['Lyric']
        artist = row['Artist']
        title = row['Title']
        traning(lyrisfile, bert_model, tokenizer, model, diffusionModel, title, artist, 0.0001)
    model.save_Model(f"./DoneModel.pt")
    print('done')
if __name__ == "__main__":
    main()