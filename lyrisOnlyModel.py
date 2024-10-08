import torch
import torch.nn as nn
import clip
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel
from diffusers import StableDiffusionPipeline, DDPMScheduler
import preprocess
import pretty_midi
import sys
import os
from datetime import datetime
device = "cuda" if torch.cuda.is_available() else "cpu"


def getTOKEN_VECTOR(midi):
    midi_obj = preprocess.miditoolkit.midi.parser.MidiFile(midi)
    encoding = preprocess.MIDI_to_encoding(midi_obj)
    return torch.tensor(encoding, dtype=torch.Long, device=device)
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
class MusicBERT2DiffusionAdapterWithCLIP(nn.Module):
    

    def __init__(self, hidden_dim=128, embedding_dim=64, bert_dim = 768,clip_model_name='ViT-B/32'):
        super(MusicBERT2DiffusionAdapterWithCLIP, self).__init__()
        # latent_dim of picture is 4*64*64
        # MusicBERT to Diffusion linear layer
        latent_dim=16384
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bert_dim = bert_dim
        self.latent_dim = latent_dim
        
        # embedding layer
        self.embedding = nn.Embedding(bert_dim, embedding_dim)
        # RNN
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # full connect
        self.linear = nn.Linear(hidden_dim, latent_dim)
        
        # load clip model as error function
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
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
        embedded_seq = self.embedding(lyris_vector)
        rnn_output, (h_n, c_n) = self.rnn(embedded_seq)
        latent_vector = self.fc(h_n[-1])
        latent_vector = latent_vector.view(-1, 4, 64, 64)
        return latent_vector
    def get_diffusion_input(self, lyris_vector):
        diffusion_input = self.forward(lyris_vector)
        return diffusion_input
    
    def calculate_similarity_loss(self, lyris_vector, images):
        music_clip_embeds = self.clip_model.encode_text(lyris_vector)  # [batch_size, clip_embed_dim]
        image_clip_embeds = self.clip_model.encode_image(images)  # [batch_size, clip_embed_dim]
        similarity_loss = 1 - cosine_similarity(music_clip_embeds, image_clip_embeds).mean()

        return similarity_loss
    def load_model(model_path):
        model = torch.load(model_path)
        model.eval()  # 設置為評估模式
        return model
    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)



def traning(lyrisfile:str, midifile:str,musicBert,BERT, musicTokenizer, bertTokenizer, Adaptermodel:MusicBERT2DiffusionAdapterWithCLIP, diffusionModel):
    # Initialize the model
    # note_sequence = midi_to_note_sequence(midifile)
    # tokenization the note
    lyrisinputs = bertTokenizer(lyrisfile, return_tensors="pt", padding=True, truncation=True, max_length=Adaptermodel.bert_dim)
    
    with torch.no_grad():
        lyris_vector = BERT(**lyrisinputs).last_hidden_state[:, 0, :]
    # Get diffusion input
    diffusion_input = Adaptermodel.get_diffusion_input(lyris_vector)
    print("Diffusion Input Shape:", diffusion_input.shape)
    diffusion_input = diffusion_input.view(diffusion_input.size(0), -1) 
    # Generate images using the diffusion model
    images = diffusionModel(prompt=midifile.split('.')[0], latents=diffusion_input).images
    # Calculate similarity loss
    similarity_loss = Adaptermodel.calculate_similarity_loss(lyris_vector, images)
    similarity_loss.backward()
    print("Similarity Loss:", similarity_loss.item())
    print('saving model')
    nowtime = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./savePoint/model_{nowtime}.pt"
    Adaptermodel.save_model(model_path)
    
def predict(lyrisfile:str, midifile:str,musicBert,BERT, musicTokenizer, bertTokenizer, Adaptermodel:MusicBERT2DiffusionAdapterWithCLIP, diffusionModel):
    # Initialize the model
    lyrisinputs = bertTokenizer(lyrisfile, return_tensors="pt", padding=True, truncation=True, max_length=Adaptermodel.bert_dim)
    
    with torch.no_grad():
        lyris_vector = BERT(**lyrisinputs).last_hidden_state[:, 0, :]
    # Get diffusion input
    diffusion_input = Adaptermodel.get_diffusion_input(lyris_vector)
    print("Diffusion Input Shape:", diffusion_input.shape)
    diffusion_input = diffusion_input.view(diffusion_input.size(0), -1) 
    # Generate images using the diffusion model
    images = diffusionModel(prompt=midifile.split('.')[0], latents=diffusion_input).images
    # Calculate similarity loss
    similarity_loss = Adaptermodel.calculate_similarity_loss(lyris_vector, images)
    print("Similarity Loss:", similarity_loss.item())
    return images, similarity_loss
def main():
    print(f'start with device {device}')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    model = MusicBERT2DiffusionAdapterWithCLIP()
    diffusionModelName = "CompVis/stable-diffusion-v1-4"
    diffusionModel  = StableDiffusionPipeline.from_pretrained(diffusionModelName,
                                               variant="fp16", torch_dtype=torch.float16)
    print('done init')
    
    # Move models to the appropriate device
    bert_model.to(device)
    model.to(device)
    diffusionModel.to(device)
    bert_model.eval()
    diffusionModel.eval()
    # Directory containing MIDI files
    lyris_dir = "./traningData/lyris"

    # Get list of all MIDI files in the directory
    lyrisfiles = [os.path.join(lyris_dir, f) for f in os.listdir(lyris_dir) if f.endswith('.txt')]
    for lyrisfile in lyrisfiles:
        traning(lyrisfile, bert_model, music_tokenizer, tokenizer, model, diffusionModel)
    print('done')
if __name__ == "__main__":
    main()