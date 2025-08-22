
from fairseq.models.roberta import RobertaModel
# from fairseq.models.transformer import TransformerModel
import os
import io
import zipfile
import traceback
import miditoolkit
import random
import time
import math
import signal
import hashlib
from multiprocessing import Pool, Lock, Manager
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
from muzic.musicbert.musicbert import *
from music21 import *
from tqdm import tqdm
from enum import Enum
from itertools import chain
from music21 import *
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import accelerate # Required by diffusers
from PIL import Image
import matplotlib.pyplot as plt

from datetime import datetime
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
import torch.nn as nn
diffusers_available = True
us = environment.UserSettings()
us['musescoreDirectPNGPath'] = './mscore'
us['directoryScratch'] = './tmp'
torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
bar_max = 256
pos_resolution = 16  

velocity_quant = 4
tempo_quant = 12
min_tempo = 16
max_tempo = 256
max_ts_denominator = 6
max_notes_per_bar = 2
beat_note_factor = 4
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16
sample_len_max = 1000 
max_inst = 127
max_pitch = 127
duration_max = 8

lock_file = Lock()
lock_write = Lock()
lock_set = Lock()
manager = Manager()
midi_dict = manager.dict()

ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)


def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def e2t(x):
    return ts_list[x]

def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]
def v2e(x):
    return x // velocity_quant

def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e

def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator


def MIDI_to_encoding(midi_obj):
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    notes_start_pos = [time_to_pos(j.start)
                       for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    max_pos = min(max(notes_start_pos) + 1, trunc_pos)
    pos_to_info = [[None for _ in range(4)] for _ in range(
        max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
    start_distribution = [0] * pos_resolution
    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]
            encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                             1 if inst.is_drum else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), v2e(note.velocity), info[1], info[3]))
    if len(encoding) == 0:
        return list()
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                          math.log2((x / tot)) for x in start_distribution))
    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)
    encoding.sort()
    return encoding




def encoding_to_str(e, bar_max = bar_max):
    bar_index_offset = 0
    p = 0
    tokens_per_note = 8
    return ' '.join((['<s>'] * tokens_per_note)
                    + ['<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) for i in e[p: p +
                                                                                            sample_len_max] if i[0] + bar_index_offset < bar_max for j, k in enumerate(i)]
                    + (['</s>'] * (tokens_per_note
                                   - 1)))   # 8 - 1 for append_eos functionality of binarizer in fairseq
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print('loading model and data')

roberta_base = MusicBERTModel.from_pretrained('.', 
    checkpoint_file = 'muzic/musicbert/checkpoints/checkpoint_last_musicbert_base_w_genre_head.pt',
    # user_dir='musicbert'    # activate the MusicBERT plugin with this keyword
)
samp = roberta_base.model.encoder.sentence_encoder
print(samp)
del samp

samp = roberta_base.model.encoder.lm_head
print(samp)
del samp

roberta_base.cuda()
roberta_base.eval()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Move to GPU if available
text_bert_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(text_bert_device)
bert_model.eval() # Set to evaluation mode
print(f"Text BERT model loaded successfully on {text_bert_device}.")
text_bert_loaded = True

# model_id = "stabilityai/stable-diffusion-2-1-base" # Or try 2.1 base

# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32, # Use float16 on GPU
#     # Add use_auth_token=True or your HF token if needed for gated models
# )

# # Use a potentially faster/better scheduler
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.return_dict = True
# pipe.vae.register_to_config(return_dict=True)
# pipe = pipe.to(device)
# # If low GPU memory, enable attention slicing
# # pipe.enable_attention_slicing()

# sd_pipe = pipe # Assign to global variable
# stable_diffusion_loaded = True
# print(f"Stable Diffusion pipeline ({model_id}) loaded successfully.")

def parse_midi_file(sample_midi_path: str):
    midi_obj = miditoolkit.midi.parser.MidiFile(sample_midi_path)
    midi_name = sample_midi_path.split('/')[-1].split('.')[0]
    return midi_obj, midi_name
def get_bar_idx(octuple_encoding, bar):
    max_bars = octuple_encoding[-1][0] 

    if(bar > max_bars):
      print('starting bar greater than total no. of bars')
      return 
    
    bars = list(zip(*octuple_encoding))[0]
    
    return bars.index(bar)

def shift_bar_to_front(octuple_encoding):
    min_bar = octuple_encoding[0][0]

    for index, oct in enumerate(octuple_encoding): 
        oct_lst = list(oct)
        oct_lst[0] -= min_bar
        octuple_encoding[index] = tuple(oct_lst)

    return octuple_encoding 


#Masks every element of octuples with `program` except the program entry, predicting masks on this leads to remixed instrument

#program: instrument ID (https://jazz-soft.net/demo/GeneralMidi.html)
#octuplemidi_token_encoding: like torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, ......., 2, 2, 2, 2, 2, 2, 2, 2])
#percentage_mask: how much percentage of notes of `program` instrument are to be masked
#replacement_program: the `replacement_program` instrument that will replace the masked octuples, 
#                       perform mask prediction to predict notes of `replacement_program` in place
#mask_attribs: Only these elements of octuples will be masked from the chosen octuples that are to be masked,
# e.g: If `mask_attribs` = [0,1,3,4,5,6,7], everything except `program` will be masked in octuples, similarly,
# If `mask_attribs` = [3], only `pitch` will be masked in chosen octuples


# The tokens should be in order (`0-bar`, `1-position`, `2-instrument`, `3-pitch`, `4-duration`, `5-velocity`, `6-timesig` , `7-tempo`) so we switch temperature value accordingly
# Limit to some specific fields such as pitch temp, duration temp, velocity temp, instrument temp  

#Helper function for OCTUPLE_MODE and MULTI_OCTUPLE_MODE
# https://www.geeksforgeeks.org/break-list-chunks-size-n-python/

#Predict missing masks in sequence from left to right

'''
NOTE: Predicts ONLY the masked octuples provided in `masked_octuples` if `prediction_mode` is NOT Vanilla
Else if prediction_mode is Vanilla, it predicts all the masks in the input `octuplemidi_token_encoding`
'''

#octuplemidi_token_encoding: of the format torch.Tensor([0,0,0,0,0,0,0,0, ..........., 2,2,2,2,2,2,2,2]), where 0 is label_dict.bos_idx & 2 is label_dict.eos_idx
#prediction_mode: decides the speed of the mask prediction
#mask_attribs: decides which of the (`bar`, `position`, `instrument`, `pitch`, `duration`, `velocity`, `timesig` , `tempo`) are masked

#masked_octuples: List of the octuple indices in `octuplemidi_token_encoding` that are masked, note that the element indices for `bar` field of these elements would be (mask_octuple_idxs * 8)

GM_INSTRUMENT_MAP = {
    0: 'Acoustic Grand Piano', 1: 'Bright Acoustic Piano', 2: 'Electric Grand Piano',
    3: 'Honky-tonk Piano', 4: 'Electric Piano 1', 5: 'Electric Piano 2', 6: 'Harpsichord',
    7: 'Clavinet', 8: 'Celesta', 9: 'Glockenspiel', 10: 'Music Box', 11: 'Vibraphone',
    12: 'Marimba', 13: 'Xylophone', 14: 'Tubular Bells', 15: 'Dulcimer', 16: 'Drawbar Organ',
    17: 'Percussive Organ', 18: 'Rock Organ', 19: 'Church Organ', 20: 'Reed Organ',
    21: 'Accordion', 22: 'Harmonica', 23: 'Tango Accordion', 24: 'Acoustic Guitar (nylon)',
    25: 'Acoustic Guitar (steel)', 26: 'Electric Guitar (jazz)', 27: 'Electric Guitar (clean)',
    28: 'Electric Guitar (muted)', 29: 'Overdriven Guitar', 30: 'Distortion Guitar',
    31: 'Guitar Harmonics', 32: 'Acoustic Bass', 33: 'Electric Bass (finger)',
    34: 'Electric Bass (pick)', 35: 'Fretless Bass', 36: 'Slap Bass 1', 37: 'Slap Bass 2',
    38: 'Synth Bass 1', 39: 'Synth Bass 2', 40: 'Violin', 41: 'Viola', 42: 'Cello',
    43: 'Contrabass', 44: 'Tremolo Strings', 45: 'Pizzicato Strings', 46: 'Orchestral Harp',
    47: 'Timpani', 48: 'String Ensemble 1', 49: 'String Ensemble 2', 50: 'Synth Strings 1',
    51: 'Synth Strings 2', 52: 'Choir Aahs', 53: 'Voice Oohs', 54: 'Synth Choir',
    55: 'Orchestra Hit', 56: 'Trumpet', 57: 'Trombone', 58: 'Tuba', 59: 'Muted Trumpet',
    60: 'French Horn', 61: 'Brass Section', 62: 'Synth Brass 1', 63: 'Synth Brass 2',
    64: 'Soprano Sax', 65: 'Alto Sax', 66: 'Tenor Sax', 67: 'Baritone Sax', 68: 'Oboe',
    69: 'English Horn', 70: 'Bassoon', 71: 'Clarinet', 72: 'Piccolo', 73: 'Flute',
    74: 'Recorder', 75: 'Pan Flute', 76: 'Blown Bottle', 77: 'Shakuhachi', 78: 'Whistle',
    79: 'Ocarina', 80: 'Lead 1 (square)', 81: 'Lead 2 (sawtooth)', 82: 'Lead 3 (calliope)',
    83: 'Lead 4 (chiff)', 84: 'Lead 5 (charang)', 85: 'Lead 6 (voice)', 86: 'Lead 7 (fifths)',
    87: 'Lead 8 (bass + lead)', 88: 'Pad 1 (new age)', 89: 'Pad 2 (warm)', 90: 'Pad 3 (polysynth)',
    91: 'Pad 4 (choir)', 92: 'Pad 5 (bowed)', 93: 'Pad 6 (metallic)', 94: 'Pad 7 (halo)',
    95: 'Pad 8 (sweep)', 96: 'FX 1 (rain)', 97: 'FX 2 (soundtrack)', 98: 'FX 3 (crystal)',
    99: 'FX 4 (atmosphere)', 100: 'FX 5 (brightness)', 101: 'FX 6 (goblins)',
    102: 'FX 7 (echoes)', 103: 'FX 8 (sci-fi)', 104: 'Sitar', 105: 'Banjo', 106: 'Shamisen',
    107: 'Koto', 108: 'Kalimba', 109: 'Bagpipe', 110: 'Fiddle', 111: 'Shanai',
    112: 'Tinkle Bell', 113: 'Agogo', 114: 'Steel Drums', 115: 'Woodblock',
    116: 'Taiko Drum', 117: 'Melodic Tom', 118: 'Synth Drum', 119: 'Reverse Cymbal',
    120: 'Guitar Fret Noise', 121: 'Breath Noise', 122: 'Seashore', 123: 'Bird Tweet',
    124: 'Telephone Ring', 125: 'Helicopter', 126: 'Applause', 127: 'Gunshot'
}
# Assume necessary functions and variables like parse_midi_file, MIDI_to_encoding,
# encoding_to_str, roberta_base, bert_tokenizer, bert_model, device, etc., are defined elsewhere.

def extract_midi_embedding(roberta_model, roberta_label_dict,
                           octuplemidi_token_encoding: torch.Tensor,
                           pooling_strategy: str = 'mean',
                           device: torch.device = None):
    if octuplemidi_token_encoding is None or not isinstance(octuplemidi_token_encoding, torch.Tensor):
        print("Error: Input 'octuplemidi_token_encoding' must be a valid Tensor.")
        return None
    if octuplemidi_token_encoding.dim() != 1:
        print("Error: Input tensor must be 1-dimensional.")
        return None

    if device is None:
        try:
            device = next(roberta_model.parameters()).device
        except Exception:
            print("Warning: Could not determine model device, using CPU.")
            device = torch.device("cpu")

    input_tensor_for_model = octuplemidi_token_encoding.unsqueeze(0).to(device)

    if input_tensor_for_model.size(1) % 8 != 0:
        padding_length = 8 - (input_tensor_for_model.size(1) % 8)
        pad_value = roberta_label_dict.pad_index if hasattr(roberta_label_dict, 'pad_index') else roberta_label_dict.pad()
        input_tensor_for_model = torch.nn.functional.pad(input_tensor_for_model, (0, padding_length), value=pad_value)

    try:
        roberta_model.eval()
        with torch.no_grad():
            model_output_result = roberta_model.model(input_tensor_for_model, features_only=True, return_all_hiddens=False)

            # Initialize token_embeddings to None for later checks
            token_embeddings = None
            last_hidden_state_from_model = None # Used to store the main extracted feature tensor

            if isinstance(model_output_result, dict) and 'x' in model_output_result:
                last_hidden_state_from_model = model_output_result['x'] # Shape: [seq_len, batch_size, hidden_size]
                print(f"Extracted from dict['x'], shape: {last_hidden_state_from_model.shape}")
            elif isinstance(model_output_result, tuple) and len(model_output_result) > 0:
                print(f"Model output is a tuple. Length: {len(model_output_result)}")
                # Usually, the first element of the tuple is the last hidden state
                # We need to confirm its dimension and shape
                if isinstance(model_output_result[0], torch.Tensor):
                    last_hidden_state_from_model = model_output_result[0]
                    print(f"Extracted from tuple[0], shape: {last_hidden_state_from_model.shape}")
                    # For more robustness, we can iterate through the tensors in the tuple to find the one with the best matching dimension
                    # for i, item in enumerate(model_output_result):
                    #     if isinstance(item, torch.Tensor) and item.dim() == 3:
                    #         print(f"Tuple element {i} shape: {item.shape}")
                    #         # More logic can be added here to select the correct tensor, e.g., based on dimension or matching with 768
                    #         if item.shape[-1] == 768: # Assuming hidden_size is 768
                    #             last_hidden_state_from_model = item
                    #             print(f"Selected tuple element {i} as features.")
                    #             break
                else:
                    print(f"Error: First element of model output tuple is not a Tensor. Type: {type(model_output_result[0])}")
                    return None
            elif isinstance(model_output_result, torch.Tensor) and model_output_result.dim() == 3:
                last_hidden_state_from_model = model_output_result # Shape: [seq_len or batch, batch or seq_len, hidden_size]
                print(f"Extracted from direct tensor output, shape: {last_hidden_state_from_model.shape}")
            else:
                print(f"Error: Unexpected model output type: {type(model_output_result)}. Could not extract hidden states.")
                return None

            # --- Process the extracted last_hidden_state_from_model ---
            if last_hidden_state_from_model is None:
                print("Error: last_hidden_state_from_model is None after attempting extraction.")
                return None

            # Expected hidden_size is 768
            expected_hidden_size = 768
            if last_hidden_state_from_model.shape[-1] != expected_hidden_size:
                print(f"Error: Extracted features have incorrect hidden dimension: {last_hidden_state_from_model.shape[-1]}. Expected {expected_hidden_size}.")
                return None

            # Convert to [batch_size, seq_len, hidden_size]
            # Fairseq encoder usually outputs [seq_len, batch_size, hidden_size]
            if last_hidden_state_from_model.shape[0] == input_tensor_for_model.shape[1] and \
               last_hidden_state_from_model.shape[1] == input_tensor_for_model.shape[0]: # [seq_len, batch (1), hidden_size]
                token_embeddings = last_hidden_state_from_model.transpose(0, 1)
            # Some implementations might directly output [batch_size, seq_len, hidden_size]
            elif last_hidden_state_from_model.shape[0] == input_tensor_for_model.shape[0] and \
                 last_hidden_state_from_model.shape[1] == input_tensor_for_model.shape[1]: # [batch (1), seq_len, hidden_size]
                token_embeddings = last_hidden_state_from_model
            else:
                print(f"Error: Shape mismatch after feature extraction. Features shape: {last_hidden_state_from_model.shape}, Input shape for model: {input_tensor_for_model.shape}. Cannot determine correct transpose.")
                return None
            
            print(f"Shape of token_embeddings after processing and transpose (if any): {token_embeddings.shape}") # Expected: [1, seq_len, 768]


            # --- Pooling ---
            if pooling_strategy == 'cls':
                sequence_embedding = token_embeddings[:, 0, :]
            elif pooling_strategy == 'mean':
                pad_idx = roberta_label_dict.pad_index if hasattr(roberta_label_dict, 'pad_index') else roberta_label_dict.pad()
                attention_mask_for_pooling = (input_tensor_for_model != pad_idx).to(device) # [batch_size, sequence_length]
                
                # Ensure the sequence length of attention_mask_for_pooling matches that of token_embeddings
                if attention_mask_for_pooling.shape[1] != token_embeddings.shape[1]:
                    print(f"Warning: Sequence length mismatch between attention mask ({attention_mask_for_pooling.shape[1]}) and token_embeddings ({token_embeddings.shape[1]}). This might lead to incorrect pooling.")
                    # You might need to readjust, truncate, or pad attention_mask_for_pooling based on token_embeddings.shape[1]
                    # For simplicity, this is not handled here, but it should be noted in practical applications

                input_mask_expanded = attention_mask_for_pooling.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled_embeddings = sum_embeddings / sum_mask
                sequence_embedding = mean_pooled_embeddings
            else:
                print(f"Error: Invalid pooling_strategy '{pooling_strategy}'. Choose 'mean' or 'cls'.")
                return None

            return sequence_embedding.cpu()

    except Exception as e:
        print(f"Error during MIDI embedding extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def midiEmbeding(midi_path):
    midi_obj, midi_name = parse_midi_file(midi_path)
    if midi_obj is None:
        print(f"MIDI parsing failed for {midi_path}.") # Add some logging
        return None # Modification: if parsing fails, return None
    # print(f"> Parsed MIDI: {midi_name} ({len(midi_obj.instruments)} tracks)") # Can be kept or removed

    octuple_encoding_list = MIDI_to_encoding(midi_obj)
    if not octuple_encoding_list:
        print(f"MIDI to encoding failed or resulted in empty sequence for {midi_path}.") # Add log
        return None # Modification: if encoding fails, return None
        # raise ValueError("MIDI to encoding failed or resulted in empty sequence.")
    # Optional: Apply bar shifting/filtering if necessary
    # starting_bar = 0
    # idx = get_bar_idx(octuple_encoding_list, starting_bar)
    # if idx is not None: octuple_encoding_list = octuple_encoding_list[idx:]
    # octuple_encoding_list = shift_bar_to_front(octuple_encoding_list)

    # Truncate if needed (MusicBERT has sequence length limits)
    # The limit is often 512 or 1024 *tokens*, not octuples.
    # Check model's max positions: roberta_base.model.max_positions()
    # Since each octuple is 8 tokens, limit octuples accordingly.
    max_octuples = sample_len_max
    if len(octuple_encoding_list) > max_octuples:
        # print(f"> Truncating octuple sequence from {len(octuple_encoding_list)} to {max_octuples}")
        octuple_encoding_list = octuple_encoding_list[:max_octuples]

    octuple_midi_str = encoding_to_str(octuple_encoding_list)
    if not octuple_midi_str:
        print(f"Encoding to string failed for {midi_path}.") # Add log
        return None # Modification: if conversion fails, return None

    # d. Tokenize the string using MusicBERT's dictionary
    label_dict = roberta_base.task.label_dictionary
    octuple_midi_tokenized = label_dict.encode_line(
        octuple_midi_str,
        append_eos=False, # encoding_to_str should have already added BOS/EOS (confirm this)
        add_if_not_exist=False
    )

    # e. Extract embedding
    midi_embedding = extract_midi_embedding(
        roberta_base,
        label_dict,
        octuple_midi_tokenized, # Pass the 1D tokenized tensor
        pooling_strategy='mean', # or 'cls'
        device=device # global device
    )

    if midi_embedding is not None:
        print(f"> Successfully extracted MIDI embedding (shape: {midi_embedding.shape})")
    else:
        print("> Failed to extract MIDI embedding.")
    return midi_embedding

def get_text_embedding(texts, tokenizer, model, device):
    """
    Generates text embeddings using a BERT model with mean pooling.

    Args:
        texts (str or list[str]): Input text(s).
        tokenizer: Pre-loaded BERT tokenizer.
        model: Pre-loaded BERT model.
        device: Torch device (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: Embeddings tensor of shape (batch_size, hidden_size).
                      Returns None if inputs are invalid or model not loaded.
    """
    if not text_bert_loaded or tokenizer is None or model is None:
        print("Error: Text BERT model/tokenizer not loaded. Cannot generate embeddings.")
        return None
    if not texts:
        print("Warning: Empty input text list provided.")
        return None

    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    try:
        # Tokenize input texts
        encoded_input = tokenizer(
            texts,
            padding=True,      # Pad sequences to max length in batch
            truncation=True,   # Truncate sequences exceeding max model length
            return_tensors='pt', # Return PyTorch tensors
            max_length=512     # Standard BERT max length
        ).to(device)

        # Get model output (no gradients needed)
        with torch.no_grad():
            model_output = model(**encoded_input)

        # --- Mean Pooling Calculation ---
        # Get last hidden states: (batch_size, sequence_length, hidden_size)
        last_hidden_state = model_output.last_hidden_state
        # Get attention mask: (batch_size, sequence_length)
        attention_mask = encoded_input['attention_mask']

        # Expand attention mask to match hidden state dimensions: (batch_size, seq_len, 1)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # Sum embeddings for non-padding tokens (mask out padding)
        # Element-wise multiplication zeros out padding embeddings
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)

        # Sum the attention mask to get the count of non-padding tokens per sequence
        # Add small epsilon for numerical stability (clamp ensures >= 1e-9)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Calculate the mean (divide sum of embeddings by count of tokens)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        return mean_pooled_embeddings # Shape: (batch_size, hidden_size)

    except Exception as e:
        print(f"Error during text embedding generation: {e}")
        print(texts)
        traceback.print_exc()
        return None

# class Decoder(nn.Module):
#     def __init__(self, embedding_dim, image_channels=3):
#         super().__init__()
#         self.token_pool = nn.Sequential(
#             nn.Conv1d(embedding_dim, 512, kernel_size=3, padding=1),  # (B, 512, 77)
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1),  # (B, 512, 1)
#             nn.Flatten()  # (B, 512)
#         )
#         self.model = nn.Sequential(
#             nn.Linear(embedding_dim, 8*8*512),
#             nn.ReLU(),
#             nn.Unflatten(1, (512, 8, 8)),
#             nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 16x16
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32x32
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64x64
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, image_channels, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # (B, 512, 77)
#         x = self.token_pool(x)  # (B, 512)
#         return self.model(x)
def load_fuse_model(timestamp, model_class, *model_args, **model_kwargs):
    model_path = f'./generated_images/{timestamp}/fuse_model.pth'
    model = model_class(*model_args, **model_kwargs)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    print("Fuse model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}, mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    return model

# Example usage:
timestamp = '20250606_015613'
# You must define the correct model class and its arguments here.
# For example, if your fuse model is a nn.Module with two inputs concatenated:
class FuseModel(torch.nn.Module):
    def __init__(self, input_dim = 1536, output_dim= 1024, hidden_dim=2048):
        super(FuseModel, self).__init__()
        
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, midi_embed, text_embed):
        x = torch.cat((midi_embed, text_embed), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

model = load_fuse_model(timestamp, FuseModel, input_dim=1536, output_dim=512, hidden_dim=2048)
model = model.to(device)
model.eval()
dataset = pd.read_csv('./kaggleData/small_song_lyrics.csv')
embeddings = []
image_paths = []
target_paths = []
error = []
# for name, param in model.named_parameters():
#     print(name, torch.mean(param).item(), torch.std(param).item())
# exit()
# print('loading fuse embedding')
for index, row in  tqdm(dataset.iterrows(), total=len(dataset), desc="Loading fuse embeddings"):
    artist = row['Artist']
    title = row['Title']
    title = title.replace('/', '_')
    title = title.replace('.', '_')
    artist = artist.replace('/', '_')
    artist = artist.replace('.', '_')
    lyric = row['Lyric']        # Assuming the dataset has a column named 'lyrics'
    midi_path = f'./trainingData/midi/{artist}-{title}.mid'  # Assuming the dataset has a column named 'midi_path'
    image_path = f'./trainingData/thumbnails/{artist}-{title}.jpg'
    embed_save_dir = f'./generated_images/{timestamp}/embeddings/'
    embed_save_path = os.path.join(embed_save_dir, f'{artist}-{title}.pt')
    if os.path.exists(embed_save_path):
        # If embedding file exists, load it directly
        prompt_embeds_final = torch.load(embed_save_path)
        print(prompt_embeds_final)
        embeddings.append(prompt_embeds_final)
        image_paths.append(image_path)
        target_paths.append(f'./generated_images/{timestamp}/thumbnails/{artist}-{title}.jpg')
        continue
    if not os.path.exists(midi_path) or not os.path.exists(image_path):
        # print(f'"{midi_path}" or "{image_path}" does not exist, skipping...')
        error.append(f"{artist}-{title}")
        continue
    if not isinstance(lyric, str) or not lyric.strip():
        # print(f'Lyric for "{midi_path}" is empty, skipping...')
        error.append(f"{artist}-{title}")
        continue
    if not os.path.exists(midi_path):
        # print(f'"{midi_path}" not exist')
        error.append(f"{artist}-{title}")
        continue
    midi_embeding = midiEmbeding(midi_path)
    if midi_embeding is None:
        # print(f'"{midi_path}" error')
        error.append(f"{artist}-{title}")
        continue
    text_embeddings = get_text_embedding(lyric, bert_tokenizer, bert_model, text_bert_device)
    if midi_embeding.dim() == 1: midi_embeding = midi_embeding.unsqueeze(0)
    if text_embeddings.dim() == 1: text_embeddings = text_embeddings.unsqueeze(0)
    midi_embeding = midi_embeding.to(device)
    text_embeddings = text_embeddings.to(device)
    with torch.no_grad():
        fused_embedding = model(midi_embeding, text_embeddings).detach()
    sequence_length = 77 # Standard for CLIP-based SD models
    prompt_embeds_final = fused_embedding.unsqueeze(1)
    prompt_embeds_final = prompt_embeds_final.repeat(1, sequence_length, 1)
    # print(f'sd_pipe.device={sd_pipe.device}, sd_pipe.text_encoder.dtype={sd_pipe.text_encoder.dtype}')
    prompt_embeds_final = prompt_embeds_final.to('cuda:0', dtype=torch.float32)
    # print(text_embeddings)
    # print(midi_embeding)
    # print(fused_embedding)
    embeddings.append(prompt_embeds_final)
    # Save the prompt_embeds_final tensor to a file for later use
    embed_save_dir = f'./generated_images/{timestamp}/embeddings/'
    os.makedirs(embed_save_dir, exist_ok=True)
    embed_save_path = os.path.join(embed_save_dir, f'{artist}-{title}.pt')
    torch.save(prompt_embeds_final.cpu(), embed_save_path)
    image_paths.append(image_path)
    target_paths.append(f'./generated_images/{timestamp}/thumbnails/{artist}-{title}.jpg')    
INPUT_C_EMB = 1   
INPUT_H_EMB = 77  
INPUT_W_EMB = 512 
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageEmbeddingDataset(Dataset):
    def __init__(self, embeddings, image_paths, target_size, image_channels):
        self.embeddings = embeddings
        self.image_paths = image_paths
        self.target_size = target_size
        self.image_channels = image_channels
        
        transform_list = [
            transforms.Resize(self.target_size),
            transforms.ToTensor() # Converts PIL Image (H, W, C) or (H, W) to Tensor (C, H, W) and normalizes to [0,1]
        ]
        # If you want to normalize to [-1, 1] (to match nn.Tanh()), you can add:
        # transforms.Normalize(mean=[0.5]*self.image_channels, std=[0.5]*self.image_channels)
        self.transform = transforms.Compose(transform_list)


    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path)
            if self.image_channels == 1 and image.mode != 'L':
                image = image.convert('L')
            elif self.image_channels == 3 and image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = self.transform(image)
            return embedding, image
        except FileNotFoundError:
            print(f"Error: Image not found {img_path} at index {idx}")
            # Return dummy data or skip, here we choose to return the first sample (requires more robust handling)
            # A better approach is to filter out invalid paths during initialization
            if idx > 0:
                return self.__getitem__(0)
            else: # If the first one cannot be found, raise an error or return null
                 raise FileNotFoundError(f"Cannot load image {img_path}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            if idx > 0:
                return self.__getitem__(0)
            else:
                 raise RuntimeError(f"Cannot load image {img_path}: {e}")

# Split the data
train_embeddings, val_embeddings, train_paths, val_paths = train_test_split(
    embeddings, image_paths, test_size=0.2, random_state=42
)

if not train_paths or not val_paths:
    print("Error: Training or validation set is empty after splitting the data. Please check the original data size.")
    exit()

train_dataset = ImageEmbeddingDataset(train_embeddings, train_paths, (IMAGE_HEIGHT, IMAGE_WIDTH), IMAGE_CHANNELS)
val_dataset = ImageEmbeddingDataset(val_embeddings, val_paths, (IMAGE_HEIGHT, IMAGE_WIDTH), IMAGE_CHANNELS)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, drop_last=True)

# Check the shape of a data batch
try:
    sample_embeddings, sample_images = next(iter(train_loader))
    print(f"Sample embeddings batch shape: {sample_embeddings.shape}") # (BATCH_SIZE, EMBEDDING_DIM)
    print(f"Sample images batch shape: {sample_images.shape}")       # (BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
except Exception as e:
    print(f"Error creating DataLoader or getting a sample: {e}")
    print("Please ensure __getitem__ and image paths in ImageEmbeddingDataset are correct.")
    exit()

# --- 3. Build the Decoder Model (`nn.Module`) ---
class Decoder(nn.Module):
    def __init__(self, input_c_in, input_h_in, input_w_in, # Embedding input shape
                 target_channels, target_height, target_width): # Output image shape
        super(Decoder, self).__init__()
        self.target_channels = target_channels
        self.target_height = target_height
        self.target_width = target_width

        flattened_input_dim = input_c_in * input_h_in * input_w_in
        self.initial_h = target_height // 8
        self.initial_w = target_width // 8
        self.initial_filters = 256 # Number of channels at the start of transpose convolution
        # print(f"DEBUG: input_c_emb = {input_c_in}")
        # print(f"DEBUG: input_h_emb = {input_h_in}")
        # print(f"DEBUG: input_w_emb = {input_w_in}")
        # print(f"DEBUG: flattened_input_dim = {flattened_input_dim}")
        # print(f"DEBUG: target_height = {target_height}")
        # print(f"DEBUG: target_width = {target_width}")
        # print(f"DEBUG: self.initial_h = {self.initial_h}")
        # print(f"DEBUG: self.initial_w = {self.initial_w}")
        # print(f"DEBUG: self.initial_filters = {self.initial_filters}") 
        # linear_out_dim = self.initial_h * self.initial_w * self.initial_filters
        # print(f"DEBUG: Calculated linear_out_dim (for nn.Linear out_features) = {linear_out_dim}")
        # print(f"DEBUG: Calculated linear_in_dim (for nn.Linear in_features) = {flattened_input_dim}")
        # num_parameters_linear = linear_out_dim * flattened_input_dim
        # memory_bytes_linear = num_parameters_linear * 4 # Assuming float32
        # print(f"DEBUG: Estimated parameters for Linear layer: {num_parameters_linear}")
        # print(f"DEBUG: Estimated memory for Linear layer weights: {memory_bytes_linear / (1024**3):.2f} GB")
        if self.initial_h == 0 or self.initial_w == 0:
            raise ValueError("Target image size is too small or initial_h/w calculation resulted in 0.")

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_input_dim, self.initial_h * self.initial_w * self.initial_filters),
            nn.BatchNorm1d(self.initial_h * self.initial_w * self.initial_filters), 
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(self.initial_filters, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, target_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_emb): # x_emb shape: (Batch, C_emb, H_emb, W_emb)
        x = self.fc(x_emb)
        # Reshape to (Batch, initial_filters, initial_h, initial_w) for deconv layers
        x = x.view(-1, self.initial_filters, self.initial_h, self.initial_w)
        x = self.deconv_layers(x)
        return x

decoder = Decoder(
    input_c_in=INPUT_C_EMB, input_h_in=INPUT_H_EMB, input_w_in=INPUT_W_EMB,
    target_channels=IMAGE_CHANNELS, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH
).to(DEVICE)
print(decoder)

# Test the model's forward pass (optional)
# test_input = torch.randn(BATCH_SIZE, EMBEDDING_DIM).to(DEVICE)
# test_output = decoder(test_input)
# print(f"Decoder output shape: {test_output.shape}") # Expected: (BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)


# --- 4. Define Loss Function and Optimizer ---
criterion = nn.L1Loss() # or nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# --- 5. Train the Model ---
print("\nStarting training...")
losses = []
val_losses = []
for epoch in range(EPOCHS):
    decoder.train() # Set to training mode
    train_loss = 0.0
    for batch_idx, (emb, real_images) in enumerate(train_loader):
        emb = emb.to(DEVICE)
        real_images = real_images.to(DEVICE)
        if emb.dtype == torch.half: # or torch.float16
            # print(f"DEBUG: Converting embeddings from {embeddings.dtype} to torch.float32 for decoder input.")
            emb = emb.float()
        elif emb.dtype != torch.float32: # Handle other possible non-float32 types
            # print(f"WARNING: embeddings dtype is {embeddings.dtype}, converting to torch.float32.")
            emb = emb.to(torch.float32)

        optimizer.zero_grad()
        generated_images = decoder(emb)
        loss = criterion(generated_images, real_images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


        if batch_idx % 5 == 0:
             print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    decoder.eval() # Set to evaluation mode
    val_loss = 0.0
    with torch.no_grad(): # Do not calculate gradients during the validation phase
        for emb, real_images in val_loader:
            emb = emb.to(DEVICE)
            real_images = real_images.to(DEVICE)
            if emb.dtype == torch.half: # or torch.float16
                # print(f"DEBUG: Converting embeddings from {embeddings.dtype} to torch.float32 for decoder input.")
                emb = emb.float()
            elif emb.dtype != torch.float32: # Handle other possible non-float32 types
                # print(f"WARNING: embeddings dtype is {embeddings.dtype}, converting to torch.float32.")
                emb = emb.to(torch.float32)
            generated_images = decoder(emb)
            vloss = criterion(generated_images, real_images)
            val_loss += vloss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
    losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
torch.save(decoder.state_dict(), f'./generated_images/{timestamp}/trained_decoder.pth')
print(f"Saved model at ./generated_images/{timestamp}/trained_decoder.pth")

# --- 6. (Optional) Test and Generate Images ---
print("\nDone training. Generating sample picture...")
decoder.eval() # Ensure it is in evaluation mode

if len(val_dataset) > 0:
    # Take an embedding from the validation set to generate an image
    _, original_image_tensor = val_dataset[0] # The first return value is the embedding, the second is the image
    sample_embedding_tensor = val_dataset.embeddings[0].unsqueeze(0).to(DEVICE) # Add batch dimension and move to device
    if sample_embedding_tensor.dtype == torch.half: # or torch.float16
        # print(f"DEBUG: Converting embeddings from {embeddings.dtype} to torch.float32 for decoder input.")
        sample_embedding_tensor = sample_embedding_tensor.float()
    elif sample_embedding_tensor.dtype != torch.float32: # Handle other possible non-float32 types
        # print(f"WARNING: embeddings dtype is {embeddings.dtype}, converting to torch.float32.")
        sample_embedding_tensor = sample_embedding_tensor.to(torch.float32)
    with torch.no_grad():
        generated_image_tensor = decoder(sample_embedding_tensor).squeeze(0).cpu() # Remove batch dimension and move to CPU

    # Convert tensor to numpy array for display
    # PyTorch tensor (C, H, W) -> Matplotlib (H, W, C)
    original_image_np = original_image_tensor.permute(1, 2, 0).detach().numpy()
    generated_image_np = generated_image_tensor.permute(1, 2, 0).numpy()

    # Ensure pixel values are in the [0,1] range (Sigmoid should handle this, but an extra step for safety)
    original_image_np = np.clip(original_image_np, 0, 1)
    generated_image_np = np.clip(generated_image_np, 0, 1)
    # Scale both the generated and original images to [0, 255] and convert to uint8 for visual comparison
    original_image_np_vis = (original_image_np * 255).astype(np.uint8)
    generated_image_np_vis = (generated_image_np * 255).astype(np.uint8)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    if IMAGE_CHANNELS == 1:
        plt.imshow(original_image_np.squeeze(), cmap='gray')
    else:
        plt.imshow(original_image_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Generated Image")
    if IMAGE_CHANNELS == 1:
        plt.imshow(generated_image_np.squeeze(), cmap='gray')
    else:
        plt.imshow(generated_image_np)
    plt.axis('off')
    
    output_filename_pytorch = f"./generated_images/{timestamp}/generated_vs_original_pytorch.png"
    plt.savefig(output_filename_pytorch)
    print(f"Comparison plot saved at {output_filename_pytorch}")

    # You can also save the model for later use
    
else:
    print("Validation set is empty, cannot generate a sample image.")

plt.figure(figsize=(8, 5))
plt.plot(losses, marker='o', label='Train Loss', color='blue')
plt.plot(val_losses, marker='x', label='Validation Loss', color='red')
plt.legend()
plt.title("Decoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(f'./generated_images/{timestamp}/decoder_training_loss.png')
# Save the trained decoder
torch.save(decoder.state_dict(), f'./generated_images/{timestamp}/trained_decoder.pth')
print("Decoder training complete and model saved.")
decoder.eval()

os.makedirs(f'./generated_images/{timestamp}/thumbnails/', exist_ok=True)
decoder.eval()
for i in tqdm(range(len(embeddings)), desc="Generating pictures from embeddings"):
    with torch.no_grad():
        current_embedding_tensor = embeddings[i].unsqueeze(0).to(DEVICE) # Get the embedding tensor
        target_image_path = target_paths[i]
        with torch.no_grad():
            generated_image_tensor = decoder(current_embedding_tensor).squeeze(0).cpu().clamp(0, 1)
        img_np = generated_image_tensor.permute(1, 2, 0).numpy()

        # Convert pixel values from [0, 1] back to [0, 255] and set as uint8
        img_np = (img_np * 255).astype(np.uint8)

        # Save the image using PIL
        img_pil = Image.fromarray(img_np)
        img_pil.save(target_image_path)