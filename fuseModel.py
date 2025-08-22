import os

from glob import glob
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
hyper_parameter = {
    "num_epoches" : 500,
    "lr_G" : 1e-4,
    "lr_D" : 1e-4,
    "device" : torch.device("cuda"),
    "batch_size" : 32,
    "step_size" : 5,
    "embeds_size" : 768,
    "sequence_size" : 77,
    "hidden_size" : 2048,
    "beta" : (0.5, 0.999),
    "gamma" : 0.5,
    "showLog" : True
}
class FuseModel(torch.nn.Module):
    def __init__(self, embed_dim=512, output_dim=1024, hidden_dim=2048):
        super(FuseModel, self).__init__()

        # Project midi and text to the same embedding dimension
        self.midi_proj = torch.nn.Linear(embed_dim, hidden_dim)
        self.text_proj = torch.nn.Linear(embed_dim, hidden_dim)

        # Cross-attention: midi attends to text
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Fully connected layers after attention
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, output_dim)

        self.relu = torch.nn.ReLU()

    def forward(self, midi_embed, text_embed):
        # Ensure inputs have the right shape: (batch, embed_dim) -> (batch, 1, embed_dim)
        if midi_embed.dim() == 2:
            midi_embed = midi_embed.unsqueeze(1)  # Add sequence dimension
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)  # Add sequence dimension
            
        # Shape: (batch, seq_len, embed_dim)
        midi = self.midi_proj(midi_embed)
        text = self.text_proj(text_embed)

        # Cross attention: midi queries, text as keys and values
        attended, _ = self.cross_attention(query=midi, key=text, value=text)

        # Pool over sequence dimension to get (batch, hidden_dim)
        pooled = attended.squeeze(1)  # Remove sequence dimension since we only have 1 element

        # Pass through MLP
        x = self.relu(self.fc1(pooled))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x
class Discriminator(torch.nn.Module):
    def __init__(self, embed_dim=768, sequence_length=77):
        super(Discriminator, self).__init__()
        # Input shape: (batch_size, 1, sequence_length, embed_dim)
        # We'll treat embed_dim as channels for the 1D convolution
        self.conv1 = torch.nn.Conv1d(in_channels=embed_dim, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        
        self.relu = torch.nn.LeakyReLU(0.2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        # use softmax->CE

    def forward(self, x):
        # x shape: (batch, 1, seq_len, embed_dim)

        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        else:
            raise ValueError(f"Expected input with 3 dimensions after squeeze, got {x.shape}")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Pool over the sequence dimension
        x = self.pool(x) # Shape: (batch, 128, 1)
        x = x.squeeze(2) # Shape: (batch, 128)
        
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.fc2(x)
        # x = self.softmax(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(-1)  # Ensure output shape is [batch]
        return x
def load_real_data(batch_size):
    files = glob('./input/trueEmbedding/*.pt')
    batch = []
    for path in files[:batch_size]:
        data = torch.load(path,weights_only=True)
        # Pad data if necessary to ensure shape [1, 77, 768]
        if data.shape[1] < hyper_parameter['sequence_size']:
            pad_size = hyper_parameter['sequence_size'] - data.shape[1]
            padding = torch.zeros((data.shape[0], pad_size, data.shape[2]), dtype=data.dtype, device=data.device)
            data = torch.cat([data, padding], dim=1)
        elif data.shape[1] > hyper_parameter['sequence_size']:
            data = data[:, :hyper_parameter['sequence_size'], :]
        # print(data.shape)
        data = data.view(1, -1)
        batch.append(data)
    return torch.cat(batch, dim=0).to(hyper_parameter['device'])
def load_fake_data(batch_size):
    files = glob('./input/generateEmbedding/*.pt')
    batch = []
    for path in files[:batch_size]:
        data = torch.load(path,weights_only=True)
        data = data.view(1, -1)
        
        batch.append(data)
    return torch.cat(batch, dim=0).to(hyper_parameter['device'])

class musicDataset(Dataset):
    def __init__(self, midi_embeddings, text_embeddings, true_embeddings):
        self.midi_embeddings = midi_embeddings
        self.text_embeddings = text_embeddings
        self.y = true_embeddings
    def __len__(self):
        return len(self.midi_embeddings)
    def __getitem__(self, index):
        x = (self.midi_embeddings[index], self.text_embeddings[index])
        y = self.y[index]
        return x, y
    
discriminator = Discriminator(embed_dim=hyper_parameter['embeds_size'], 
                              sequence_length=hyper_parameter['sequence_size']).to(hyper_parameter['device'])
fuse_model = FuseModel(embed_dim=hyper_parameter['embeds_size'], 
                       output_dim=hyper_parameter['embeds_size']*hyper_parameter['sequence_size'], 
                       hidden_dim=hyper_parameter['hidden_size']).to(hyper_parameter['device'])
criterion = torch.nn.BCELoss()

opt_G = torch.optim.Adam(fuse_model.parameters(), lr=hyper_parameter["lr_G"], betas=hyper_parameter['beta'])
opt_D = torch.optim.Adam(discriminator.parameters(), lr=hyper_parameter["lr_D"], betas=hyper_parameter['beta'])

scheduler_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=hyper_parameter['step_size'], gamma=hyper_parameter['gamma'])
scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=hyper_parameter['step_size'], gamma=hyper_parameter['gamma'])
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

discriminator.apply(weights_init)
fuse_model.apply(weights_init)
def main():
    
    print("loading data...")
    datasetPath = './kaggleData/small_song_lyrics.csv'
    # Read the dataset as a CSV file
    dataset = pd.read_csv(datasetPath)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    
    midi_embedding_list = []
    text_embedding_list = []
    true_embedding_list = []
    for index, row in train_set.iterrows():
        artist = row['Artist']
        title = row['Title']
        title = title.replace('/', '_')
        title = title.replace('.', '_')
        artist = artist.replace('/', '_')
        artist = artist.replace('.', '_')
        midi_path = os.path.join('./input/midi', f"{artist}-{title}.pt")
        if not os.path.exists(midi_path):
            continue
        midi_embedding = torch.load(midi_path,weights_only=True)
        
        text_path = os.path.join('./input/text', f"{artist}-{title}.pt")
        if not os.path.exists(text_path):
            continue
        text_embedding = torch.load(text_path,weights_only=True)
        
        real_path = os.path.join('./input/trueEmbedding', f"{artist}-{title}.pt")
        if not os.path.exists(real_path):
            continue
        real_data = torch.load(real_path,weights_only=True)
        if real_data.shape[1] < hyper_parameter['sequence_size']:
            pad_size = hyper_parameter['sequence_size'] - real_data.shape[1]
            padding = torch.zeros((real_data.shape[0], pad_size, real_data.shape[2]), dtype=real_data.dtype, device=real_data.device)
            real_data = torch.cat([real_data, padding], dim=1)
        elif real_data.shape[1] > hyper_parameter['sequence_size']:
            real_data = real_data[:, :hyper_parameter['sequence_size'], :]
        
        midi_embedding_list.append(midi_embedding)
        text_embedding_list.append(text_embedding)
        true_embedding_list.append(real_data)
    
    
    midi_embedding_list = []
    text_embedding_list = []
    true_embedding_list = []
    for index, row in test_set.iterrows():
        artist = row['Artist']
        title = row['Title']
        title = title.replace('/', '_')
        title = title.replace('.', '_')
        artist = artist.replace('/', '_')
        artist = artist.replace('.', '_')
        midi_path = os.path.join('./input/midi', f"{artist}-{title}.pt")
        if not os.path.exists(midi_path):
            continue
        midi_embedding = torch.load(midi_path,weights_only=True)
        
        text_path = os.path.join('./input/text', f"{artist}-{title}.pt")
        if not os.path.exists(text_path):
            continue
        text_embedding = torch.load(text_path,weights_only=True)
        
        real_path = os.path.join('./input/trueEmbedding', f"{artist}-{title}.pt")
        if not os.path.exists(real_path):
            continue
        real_data = torch.load(real_path,weights_only=True)
        if real_data.shape[1] < hyper_parameter['sequence_size']:
            pad_size = hyper_parameter['sequence_size'] - real_data.shape[1]
            padding = torch.zeros((real_data.shape[0], pad_size, real_data.shape[2]), dtype=real_data.dtype, device=real_data.device)
            real_data = torch.cat([real_data, padding], dim=1)
        elif real_data.shape[1] > hyper_parameter['sequence_size']:
            real_data = real_data[:, :hyper_parameter['sequence_size'], :]
        
        midi_embedding_list.append(midi_embedding)
        text_embedding_list.append(text_embedding)
        true_embedding_list.append(real_data)
    train_dataset = musicDataset(midi_embedding_list, text_embedding_list, true_embedding_list)
    train_dataloader = DataLoader(train_dataset, batch_size=hyper_parameter['batch_size'], shuffle=True)
    
    
    test_dataset = musicDataset(midi_embedding_list, text_embedding_list, true_embedding_list)
    test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    total_loss_D = 0.0
    total_loss_G = 0.0
    lossG = []
    lossD = []
    if hyper_parameter['showLog']:
        writer = SummaryWriter(log_dir=f'./runs/{timestamp}')
    print("start training...")
    start_time= time.time()
    for epoch in range(hyper_parameter["num_epoches"]):
        training_size = 0
        elossD = 0
        elossG = 0
        for idx, data in enumerate(train_dataloader):
            # --- Train G ---
            opt_G.zero_grad()
            
            x, y = data
            midi_embeddings = x[0].to(hyper_parameter['device'])
            text_embeddings = x[1].to(hyper_parameter['device'])
            fused_embedding = fuse_model(midi_embeddings, text_embeddings)
            sequence_length = hyper_parameter['sequence_size']
            embed_dim = hyper_parameter['embeds_size']
            # If fused_embedding is [batch_size, 768*77], reshape to [batch_size, 1, 77, 768]
            if fused_embedding.dim() == 2 and fused_embedding.shape[1] == sequence_length * embed_dim:
                prompt_embedding = fused_embedding.view(len(midi_embeddings), sequence_length, embed_dim)
            else:
                # If not, try to expand/repeat to [batch_size, 1, 77, 768]
                prompt_embedding = fused_embedding.unsqueeze(1).unsqueeze(2).repeat(1, sequence_length, 1)
            fake_embeddings = prompt_embedding
            fake_pred = discriminator(fake_embeddings)
            real_labels = torch.ones(len(midi_embeddings), dtype=torch.float).to(hyper_parameter['device']) 
            loss_G = criterion(fake_pred, real_labels)
            loss_G.backward()
            opt_G.step()
            elossG += loss_G.item()
            lossG.append(loss_G.item())

            # --- Train D ---         
            opt_D.zero_grad()
            true_embeddings = y.to(hyper_parameter['device'])
            real_labels = torch.ones(len(midi_embeddings), dtype=torch.float).to(hyper_parameter['device']) 
            fake_labels = torch.zeros(len(midi_embeddings), dtype=torch.float).to(hyper_parameter['device'])
            real_pred = discriminator(true_embeddings)
            loss_real = criterion(real_pred, real_labels)
            fake_pred2 = discriminator(fake_embeddings.detach())
            loss_fake = criterion(fake_pred2, fake_labels)
            loss_D = (loss_real + loss_fake)/2
            loss_D.backward()
            opt_D.step()
            # Recompute fake_pred for generator to avoid backward-through-graph error
            # fake_pred_for_G = discriminator(fake_embeddings)
            elossD += loss_D.item()
            lossD.append(loss_D.item())
            training_size += 1
            
            if(hyper_parameter['showLog']):
                writer.add_scalars('Loss', {
                    'Discriminator': loss_D.item(),
                    'Generator': loss_G.item()
                }, epoch * len(train_set)//hyper_parameter['batch_size'] + idx)
        
        scheduler_G.step()
        scheduler_D.step()
        print(f"Epoch {epoch+1}/{hyper_parameter['num_epoches']}, D_loss: {elossD/len(train_dataloader):.4f}, G_loss: {elossG/len(train_dataloader):.4f}")
    os.makedirs(f'./models/{timestamp}', exist_ok=True)
    torch.save(fuse_model, f'./models/{timestamp}/Generator.pth')
    torch.save(discriminator, f'./models/{timestamp}/Discriminator.pth')
    print('Model saved.')
    print('Training Finished.')
    elapsed = int(time.time() - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    print('Cost Time: {:02d}h{:02d}m{:02d}s'.format(hours, minutes, seconds))
    print('start testing...')
    # TODO:
    
    print('Testing Finished.')
    
    print('start generating.')
    # Clear the generateEmbedding directory before each epoch
    generate_embedding_dir = './input/generateEmbedding'
    if os.path.exists(generate_embedding_dir):
        for f in os.listdir(generate_embedding_dir):
            file_path = os.path.join(generate_embedding_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    plt.figure(figsize=(8, 5))
    plt.plot(lossG, color='orange', label='Generator Loss')
    plt.plot(lossD, color='blue', label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./models/{timestamp}/Traceback.png')
    print(f"Training completed.")
    for index, row in dataset.iterrows():
        artist = row['Artist']
        title = row['Title']
        title = title.replace('/', '_')
        title = title.replace('.', '_')
        artist = artist.replace('/', '_')
        artist = artist.replace('.', '_')
        midi_path = os.path.join('./input/midi', f"{artist}-{title}.pt")
        if not os.path.exists(midi_path):
            continue
        midi_embedding = torch.load(midi_path,weights_only=True)
        
        text_path = os.path.join('./input/text', f"{artist}-{title}.pt")
        if not os.path.exists(text_path):
            continue
        text_embedding = torch.load(text_path,weights_only=True)

        midi_embedding = midi_embedding.to(hyper_parameter['device'])
        text_embedding = text_embedding.to(hyper_parameter['device'])
        fused_embedding = fuse_model(midi_embedding, text_embedding)
        sequence_length = hyper_parameter['sequence_size'] # Standard for CLIP-based SD models
        embed_dim = hyper_parameter['embeds_size']
        if fused_embedding.dim() == 2 and fused_embedding.shape[1] == sequence_length * embed_dim:
            prompt_embedding = fused_embedding.view(len(midi_embedding), sequence_length, embed_dim)
        else:
            prompt_embedding = fused_embedding.unsqueeze(1).unsqueeze(2).repeat(1, sequence_length, 1)
        generate_embedding_dir = './input/generateEmbedding'
        os.makedirs(generate_embedding_dir, exist_ok=True)
        generate_embedding_path = os.path.join(generate_embedding_dir, f"{artist}-{title}.pt")
        torch.save(prompt_embedding, generate_embedding_path)
if __name__ == "__main__":
    main()
