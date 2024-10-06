import torch
import torch.nn as nn
import clip
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel
from diffusers import StableDiffusionPipeline, DDPMScheduler

import pretty_midi
import sys
import os

def midi_to_note_sequence(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    note_sequence = []

    # 遍歷所有樂器和音符
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # 將 MIDI 數值轉換為音符名稱 (如 C4)
            note_name = pretty_midi.note_number_to_name(note.pitch)
            note_sequence.append(note_name)
    # 將音符序列合併為字符串，例如 "C4 E4 G4 C5"
    return " ".join(note_sequence)
class MusicBERTToDiffusionAdapterWithCLIP(nn.Module):
    def _padding(self, token_vectors):
        """
        私有函數，用於填充或截斷 token vectors
        :param token_vectors: [batch_size, seq_len, hidden_size]，來自 MusicBERT 的輸出
        :return: 填充或截斷後的 token vectors
        """
        batch_size, seq_len, hidden_size = token_vectors.size()
        
        if seq_len > self.max_length:
            token_vectors = token_vectors[:, :self.max_length, :]
        elif seq_len < self.max_length:
            padding_size = self.max_length - seq_len
            padding = torch.zeros(batch_size, padding_size, hidden_size, device=token_vectors.device)
            token_vectors = torch.cat((token_vectors, padding), dim=1)
        
        return token_vectors
    def __init__(self, hidden_size=768, max_length=1024, diffusion_input_size=256, clip_model_name='ViT-B/32'):
        """
        
        :param hidden_size: MusicBERT hidden_size（例如 768）
        :param max_length: 固定的最大序列長度（用於 padding）
        :param diffusion_input_size: Diffusion Model 的輸入大小
        :param clip_model_name: CLIP 模型的名稱（如 'ViT-B/32'）
        """
        super(MusicBERTToDiffusionAdapterWithCLIP, self).__init__()
        
        # MusicBERT to Diffusion linear layer
        self.linear = nn.Linear(hidden_size, diffusion_input_size)
        # self.linear2 = nn.Linear(diffusion_input_size, diffusion_input_size*diffusion_input_size)
        # self.linear3 = nn.Linear(diffusion_input_size*diffusion_input_size, diffusion_input_size*diffusion_input_size*diffusion_input_size)
        # self.linear4 = nn.Linear(diffusion_input_size*diffusion_input_size*diffusion_input_size, diffusion_input_size*diffusion_input_size*diffusion_input_size*3)
        
        # padding
        self.max_length = max_length
        
        # load clip model as error function
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
        self.clip_model.eval() 
        
    def get_diffusion_input(self, token_vectors):
        self._padding(token_vectors)
        # Step 2: 通過線性層進行轉換
        diffusion_input = self.linear(token_vectors)  # [batch_size, max_length, diffusion_input_size]
        return diffusion_input
    
    def calculate_similarity_loss(self, token_vectors, images):
        """
        計算音樂 token vectors 和圖片的 CLIP 相似性損失
        :param token_vectors: [batch_size, seq_len, hidden_size]，來自 MusicBERT 的輸出
        :param images: [batch_size, 3, H, W]，圖片張量
        :return: 相似性損失
        """
        
        # Step 3: 提取 MusicBERT token vector 的 CLIP 嵌入
        music_clip_embeds = self.clip_model.encode_text(token_vectors)  # [batch_size, clip_embed_dim]
        
        # Step 4: 將圖片處理並提取 CLIP 圖片嵌入
        image_clip_embeds = self.clip_model.encode_image(images)  # [batch_size, clip_embed_dim]
        
        # Step 5: 計算音樂和圖片嵌入的相似性損失
        similarity_loss = 1 - cosine_similarity(music_clip_embeds, image_clip_embeds).mean()

        return similarity_loss
    def load_model(model_path):
        model = torch.load(model_path)
        model.eval()  # 設置為評估模式
        return model
    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)



device = "cuda" if torch.cuda.is_available() else "cpu"
def traning(midifile:str,musicBert,musicTokenizer, Adaptermodel:MusicBERTToDiffusionAdapterWithCLIP, diffusionModel):
    # Initialize the model
    # note_sequence = midi_to_note_sequence(midifile)
    # tokenization the note
    inputs = musicTokenizer(midifile, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # inputs = musicTokenizer(note_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        token_vectors = musicBert(**inputs).last_hidden_state
    # Get diffusion input
    diffusion_input = Adaptermodel.get_diffusion_input(token_vectors)
    print("Diffusion Input Shape:", diffusion_input.shape)
    diffusion_input = diffusion_input.view(diffusion_input.size(0), -1) 
    # Generate images using the diffusion model
    images = diffusionModel(prompt=midifile.split('.')[0], latents=diffusion_input).images
    # Calculate similarity loss
    similarity_loss = Adaptermodel.calculate_similarity_loss(token_vectors, images)
    similarity_loss.backward()
    print("Similarity Loss:", similarity_loss.item())
def predict(midifile:str,musicBert,musicTokenizer, Adaptermodel:MusicBERTToDiffusionAdapterWithCLIP, diffusionModel):
    # Initialize the model
    note_sequence = midi_to_note_sequence(midifile)
    # tokenization the note
    inputs = musicTokenizer(note_sequence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        token_vectors = musicBert(**inputs).last_hidden_state
    # Get diffusion input
    diffusion_input = Adaptermodel.get_diffusion_input(token_vectors)
    print("Diffusion Input Shape:", diffusion_input.shape)
    diffusion_input = diffusion_input.view(diffusion_input.size(0), -1) 
    images = diffusionModel(diffusion_input)
    # Calculate similarity loss
    similarity_loss = Adaptermodel.calculate_similarity_loss(token_vectors, images)
    print("Similarity Loss:", similarity_loss.item())
    return images, similarity_loss
def main():
    print('start')
    # tokenizer = BertTokenizer.from_pretrained("ruru2701/musicbert-v1.1")
    # musicbert_model = BertModel.from_pretrained("ruru2701/musicbert-v1.1")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    musicbert_model = BertModel.from_pretrained("bert-base-uncased")
    model = MusicBERTToDiffusionAdapterWithCLIP()
    diffusionModelName = "CompVis/stable-diffusion-v-1-4"
    diffusionModel  = StableDiffusionPipeline.from_pretrained(diffusionModelName,
                                               variant="fp16", torch_dtype=torch.float16)
    print('done init')
    
    # Move models to the appropriate device
    musicbert_model.to(device)
    model.to(device)
    diffusionModel.to(device)
    musicbert_model.eval()
    diffusionModel.eval()
    # Directory containing MIDI files
    midi_dir = "./midifile"

    # Get list of all MIDI files in the directory
    midifiles = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith('.mid')]
    for midifile in midifiles:
        traning(midifile, musicbert_model, tokenizer, model, diffusionModel)
    
if __name__ == "__main__":
    main()