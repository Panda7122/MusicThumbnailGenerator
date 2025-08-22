# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from fairseq.models.roberta import RobertaModel
# from fairseq.models.transformer import TransformerModel
import os
import io
import zipfile
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
import sys
from muzic.musicbert.musicbert import *
from tqdm import tqdm
from enum import Enum
from itertools import chain
from music21 import *
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import accelerate # Required by diffusers
from PIL import Image
from datetime import datetime
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

diffusers_available = True
us = environment.UserSettings()
us['musescoreDirectPNGPath'] = './mscore'
us['directoryScratch'] = './tmp'

bar_max = 256
pos_resolution = 16  

velocity_quant = 4
tempo_quant = 12
min_tempo = 16
max_tempo = 256
duration_max = 8
max_ts_denominator = 6
max_notes_per_bar = 2
beat_note_factor = 4
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16
sample_len_max = 1000 
sample_overlap_rate = 4
ts_filter = False
pool_num = 24
max_inst = 127
max_pitch = 127
max_velocity = 127

data_zip = None
output_file = None

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


def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]


def v2e(x):
    return x // velocity_quant


def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)


def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def e2b(x):
    return 2 ** (x / tempo_quant) * min_tempo
            
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


def writer(file_name, output_str_list):
    # note: parameter "file_name" is reserved for patching
    with open(output_file, 'a') as f:
        for output_str in output_str_list:
            f.write(output_str + '\n')
            
def gen_dictionary(file_name):
    num = 0
    with open(file_name, 'w') as f:
        for j in range(bar_max):
            print('<0-{}>'.format(j), num, file=f)
        for j in range(beat_note_factor * max_notes_per_bar * pos_resolution):
            print('<1-{}>'.format(j), num, file=f)
        for j in range(max_inst + 1 + 1):
            # max_inst + 1 for percussion
            print('<2-{}>'.format(j), num, file=f)
        for j in range(2 * max_pitch + 1 + 1):
            # max_pitch + 1 ~ 2 * max_pitch + 1 for percussion
            print('<3-{}>'.format(j), num, file=f)
        for j in range(duration_max * pos_resolution):
            print('<4-{}>'.format(j), num, file=f)
        for j in range(v2e(max_velocity) + 1):
            print('<5-{}>'.format(j), num, file=f)
        for j in range(len(ts_list)):
            print('<6-{}>'.format(j), num, file=f)
        for j in range(b2e(max_tempo) + 1):
            print('<7-{}>'.format(j), num, file=f)
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

def encoding_to_MIDI(encoding):
    # TODO: filter out non-valid notes and error handling
    bar_to_timesig = [list()
                      for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = e2t(bar_to_timesig[i])
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = (i[3] - 128 if program == 128 else i[3])
        duration = get_tick(0, e2d(i[4]))
        if duration == 0:
            duration = 1
        end = start + duration
        velocity = e2v(i[5])
        midi_obj.instruments[program].notes.append(miditoolkit.containers.Note(
            start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = e2t(new_ts)
            midi_obj.time_signature_changes.append(miditoolkit.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj

def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode('ascii')).hexdigest()
    return midi_hash

def F(file_name):
    try_times = 10
    midi_file = None
    for _ in range(try_times):
        try:
            lock_file.acquire()
            with data_zip.open(file_name) as f:
                # this may fail due to unknown bug
                midi_file = io.BytesIO(f.read())
        except BaseException as e:
            try_times -= 1
            time.sleep(1)
            if try_times == 0:
                print('ERROR(READ): ' + file_name +
                      ' ' + str(e) + '\n', end='')
                return None
        finally:
            lock_file.release()
    try:
        with timeout(seconds=600):
            midi_obj = miditoolkit.midi.parser.MidiFile(file=midi_file)
        # check abnormal values in parse result
        assert all(0 <= j.start < 2 ** 31 and 0 <= j.end < 2 **
                   31 for i in midi_obj.instruments for j in i.notes), 'bad note time'
        assert all(0 < j.numerator < 2 ** 31 and 0 < j.denominator < 2 **
                   31 for j in midi_obj.time_signature_changes), 'bad time signature value'
        assert 0 < midi_obj.ticks_per_beat < 2 ** 31, 'bad ticks per beat'
    except BaseException as e:
        print('ERROR(PARSE): ' + file_name + ' ' + str(e) + '\n', end='')
        return None
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    try:
        e = MIDI_to_encoding(midi_obj)
        if len(e) == 0:
            print('ERROR(BLANK): ' + file_name + '\n', end='')
            return None
        if ts_filter:
            allowed_ts = t2e(time_signature_reduce(4, 4))
            if not all(i[6] == allowed_ts for i in e):
                print('ERROR(TSFILT): ' + file_name + '\n', end='')
                return None
        if deduplicate:
            duplicated = False
            dup_file_name = ''
            midi_hash = '0' * 32
            try:
                midi_hash = get_hash(e)
            except BaseException as e:
                pass
            lock_set.acquire()
            if midi_hash in midi_dict:
                dup_file_name = midi_dict[midi_hash]
                duplicated = True
            else:
                midi_dict[midi_hash] = file_name
            lock_set.release()
            if duplicated:
                print('ERROR(DUPLICATED): ' + midi_hash + ' ' +
                      file_name + ' == ' + dup_file_name + '\n', end='')
                return None
        output_str_list = []
        sample_step = max(round(sample_len_max / sample_overlap_rate), 1)
        for p in range(0 - random.randint(0, sample_len_max - 1), len(e), sample_step):
            L = max(p, 0)
            R = min(p + sample_len_max, len(e)) - 1
            bar_index_list = [e[i][0]
                              for i in range(L, R + 1) if e[i][0] is not None]
            bar_index_min = 0
            bar_index_max = 0
            if len(bar_index_list) > 0:
                bar_index_min = min(bar_index_list)
                bar_index_max = max(bar_index_list)
            offset_lower_bound = -bar_index_min
            offset_upper_bound = bar_max - 1 - bar_index_max
            # to make bar index distribute in [0, bar_max)
            bar_index_offset = random.randint(
                offset_lower_bound, offset_upper_bound) if offset_lower_bound <= offset_upper_bound else offset_lower_bound
            e_segment = []
            for i in e[L: R + 1]:
                if i[0] is None or i[0] + bar_index_offset < bar_max:
                    e_segment.append(i)
                else:
                    break
            tokens_per_note = 8
            output_words = (['<s>'] * tokens_per_note) \
                + [('<{}-{}>'.format(j, k if j > 0 else k + bar_index_offset) if k is not None else '<unk>') for i in e_segment for j, k in enumerate(i)] \
                + (['</s>'] * (tokens_per_note - 1)
                   )  # tokens_per_note - 1 for append_eos functionality of binarizer in fairseq
            output_str_list.append(' '.join(output_words))

        # no empty
        if not all(len(i.split()) > tokens_per_note * 2 - 1 for i in output_str_list):
            print('ERROR(ENCODE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        try:
            lock_write.acquire()
            writer(file_name, output_str_list)
        except BaseException as e:
            print('ERROR(WRITE): ' + file_name + ' ' + str(e) + '\n', end='')
            return False
        finally:
            lock_write.release()
        print('SUCCESS: ' + file_name + '\n', end='')
        return True
    except BaseException as e:
        print('ERROR(PROCESS): ' + file_name + ' ' + str(e) + '\n', end='')
        return False
    print('ERROR(GENERAL): ' + file_name + '\n', end='')
    return False


def G(file_name):
    try:
        return F(file_name)
    except BaseException as e:
        print('ERROR(UNCAUGHT): ' + file_name + '\n', end='')
        return False
    
    
def str_to_encoding(s):
    encoding = [int(i[3: -1]) for i in s.split() if 's' not in i]
    tokens_per_note = 8
    assert len(encoding) % tokens_per_note == 0
    encoding = [tuple(encoding[i + j] for j in range(tokens_per_note))
                for i in range(0, len(encoding), tokens_per_note)]
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

# Load the CLIP model and processor
clip_model_id = "openai/clip-vit-base-patch32"  # You can choose a different CLIP model if needed
clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

print(f"CLIP model ({clip_model_id}) loaded successfully on {device}.")
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
def filter_tracks(midi_obj: miditoolkit.midi.parser.MidiFile, root_folder:str = '.', 
                    cache_folder:str = "musicbert_cache", midi_name:str = "final", track_ids:list = []):
    if len(track_ids) > 0:
        new_midi_obj = miditoolkit.midi.parser.MidiFile()
        new_midi_obj.instruments = [midi_obj.instruments[i] for i in track_ids]

        new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
        new_midi_obj.time_signature_changes = midi_obj.time_signature_changes
        new_midi_obj.tempo_changes = midi_obj.tempo_changes

        midi_obj = new_midi_obj
    
    midi_obj.dump(f'{root_folder}/{cache_folder}/{midi_name}_track_filtered.mid')

    print(f"> Parsed MIDI: {midi_name}")
    print(f"> Saved Parsed MIDI as : {root_folder}/{cache_folder}/{midi_name}_track_filtered.mid")
    # playMidi(f'/content/{midi_name}_track_filtered.mid')
    return midi_obj
def cache_midi_tracks(midi_obj: miditoolkit.midi.parser.MidiFile, midi_name, root_folder = '.', 
                      cache_folder = "musicbert_cache", verbose = False):
    instrument_progs = []
    for track_idx, ins_track in enumerate(midi_obj.instruments):  
        temp_midi_obj = miditoolkit.midi.parser.MidiFile()
        temp_midi_obj.instruments = [ins_track]

        temp_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
        temp_midi_obj.time_signature_changes = midi_obj.time_signature_changes
        temp_midi_obj.tempo_changes = midi_obj.tempo_changes

        instrument_progs.append(ins_track.program)
        temp_midi_obj.dump(f'{root_folder}/{cache_folder}/{midi_name}_track_{track_idx}_prog_{ins_track.program}.mid')
  
    if verbose:
        print(f'The input MIDI has {len(midi_obj.instruments)} tracks with program IDs {instrument_progs} respectively.')

def reverse_label_dict(label_dict: fairseq.data.dictionary.Dictionary):
    return {v: k for k, v in label_dict.indices.items()}
def decode_w_label_dict(label_dict: fairseq.data.dictionary.Dictionary, octuple_midi_enc:torch.Tensor,
                        skip_masked_tokens = False):
    octuple_midi_enc_copy = octuple_midi_enc.clone().tolist()
    seq = []
    rev_inv_map = reverse_label_dict(label_dict)
    for token in octuple_midi_enc_copy:
        seq.append(rev_inv_map[token])
  
    seq_str = " ".join(seq)
  
    if skip_masked_tokens:
        seq = seq_str.split()
        masked_oct_idxs = set([(idx - idx%8) for idx, elem in enumerate(seq) if elem == '<mask>'])
  
        #Deleting Octuples with any <mask> element until none remains
        try:
            while(True):
                masked_oct_idx = seq.index('<mask>')
                masked_oct_idx = masked_oct_idx - masked_oct_idx%8
                del seq[masked_oct_idx: masked_oct_idx+8]
        except ValueError: #Error: substring not found
            pass
  
        seq_str = " ".join(seq)
  
    del octuple_midi_enc_copy
    return seq_str
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

def get_min_bar_idx_from_oct(octuple_midi_str_aslist = ("<s> "*8).split() + (" </s>"*8).split()
              , min_bar_mask: int = 0):
  
  max_bars = int(octuple_midi_str_aslist[-16][3:-1])
  # print(f'max_bars = {max_bars}')
  try:
    assert min_bar_mask <= max_bars
  except:
    raise Exception(f"The input MIDI does not have {min_bar_mask} bars, it has {octuple_midi_str_aslist[-16][3:-2]} bars")

  
  try:
    # '<0-min_bar_mask>' should be present if a note from bar `min_bar_mask` is present
    min_idx = octuple_midi_str_aslist.index(f'<0-{min_bar_mask}>')
  except:
    return get_min_bar_idx_from_oct(octuple_midi_str_aslist, min_bar_mask + 1)
  

  # #Exception, if no note from the bar `min_bar_mask` is present, program fails
  try:
    assert min_idx % 8 == 0
  except:
    raise Exception("Fatal backend error!: min_idx not a multiple of 8")

  print(f'Minimum index having {min_bar_mask} bars is {min_idx} belonging to octuple with index {int(min_idx/8)} ')

  return min_idx


#Masks every element of octuples with `program` except the program entry, predicting masks on this leads to remixed instrument

#program: instrument ID (https://jazz-soft.net/demo/GeneralMidi.html)
#octuplemidi_token_encoding: like torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, ......., 2, 2, 2, 2, 2, 2, 2, 2])
#percentage_mask: how much percentage of notes of `program` instrument are to be masked
#replacement_program: the `replacement_program` instrument that will replace the masked octuples, 
#                       perform mask prediction to predict notes of `replacement_program` in place
#mask_attribs: Only these elements of octuples will be masked from the chosen octuples that are to be masked,
# e.g: If `mask_attribs` = [0,1,3,4,5,6,7], everything except `program` will be masked in octuples, similarly,
# If `mask_attribs` = [3], only `pitch` will be masked in chosen octuples

def mask_instrument_notes_program(program: int, octuplemidi_token_encoding: torch.Tensor, \
                    label_dict: fairseq.data.dictionary.Dictionary, percentage_mask = 100,
                    replacement_program:int = None, mask_attribs = [0, 1, 3, 4, 5, 6, 7], 
                    min_bar_mask = 0, seed = 42):
  np.random.seed(seed)

  octuplemidi_token_encoding = octuplemidi_token_encoding.clone()
  rev_label_dict = reverse_label_dict(label_dict)
  octuple_midi_str_aslist = [rev_label_dict[x] for x in octuplemidi_token_encoding.tolist()]

  #Find minimum index having `positon` equal to `min_bar_mask`
  #https://stackoverflow.com/questions/2361426/get-the-first-item-from-an-iterable-that-matches-a-condition
  min_idx = get_min_bar_idx_from_oct(octuple_midi_str_aslist, min_bar_mask)
  print(min_idx)

  #Expecting soft copies to be made, i.e, changing `octuplemidi_token_encoding_mutable` also changes `octuplemidi_token_encoding`
  octuplemidi_token_encoding_mutable = octuplemidi_token_encoding[min_idx: ]
  octuple_midi_str_aslist_mutable = octuple_midi_str_aslist[min_idx: ]

  instrument_octuple_indices = [int(index/8) for index,value in enumerate(octuple_midi_str_aslist_mutable) if value == f'<2-{program}>' ]

  try:
    assert len(instrument_octuple_indices) > 0
  except:
    raise Exception(f"No notes found with program = {program}")

  print(f'Found {len(instrument_octuple_indices)} octuples with program = {program}')
  print(f'Choosing {int(len(instrument_octuple_indices) * (percentage_mask/100) )} octuples to mask....')

  if percentage_mask <= 100 and percentage_mask >= 0:
    masked_octs = np.random.choice( a =  instrument_octuple_indices , \
                               size = int( len(instrument_octuple_indices) * (percentage_mask/100) ), \
                               replace = False)
    
    masked_octs = list(masked_octs)
    masked_octs.sort(reverse = False)
    
    #Prints octuple indices valid for original input `octuplemidi_token_encoding` and NOT `octuplemidi_token_encoding_mutable`
    masked_octs_orig = [( int(min_idx/8) + x ) for x in masked_octs]
    print(f'Masking octuple numbers: { masked_octs_orig}')

    mask_idx = label_dict.index('<mask>')

    replacement_program_idx = None
    if replacement_program is not None:
      replacement_program_idx = label_dict.index(f'<2-{replacement_program}>')

    for masked_oct in masked_octs:

      octuplemidi_token_encoding_mutable.index_fill_(0, torch.tensor( masked_oct * 8 + mask_attribs ), mask_idx)

      if replacement_program is not None:
        octuplemidi_token_encoding_mutable.index_fill_(0, torch.tensor( [
                      masked_oct * 8 + 2
                      ]) , replacement_program_idx)
      # octuplemidi_token_encoding[ masked_oct * 8: (masked_oct + 1)*8 ] = mask_idx
  else:
    raise IndexError

  #Expecting `octuplemidi_token_encoding` to have changed when we changed `octuplemidi_token_encoding_mutable` above

  octuplemidi_token_encoding[min_idx: ] = octuplemidi_token_encoding_mutable

  return octuplemidi_token_encoding, masked_octs_orig

BAR_START = "<0-0>"
BAR_END = "<0-255>"

POS_START = "<1-0>"
POS_END = "<1-127>"

INS_START = "<2-0>"
INS_END = "<2-127>"

PITCH_START = "<3-0>"
PITCH_END = "<3-255>"

DUR_START = "<4-0>"
DUR_END = "<4-127>"

VEL_START = "<5-0>"
VEL_END = "<5-31>"

SIG_START = "<6-0>"
SIG_END = "<6-253>"

TEMPO_START = "<7-0>"
TEMPO_END = "<7-48>"

SPECIAL_TOKENS = ['<mask>', '<s>', '<pad>', '</s>', '<unk>']

def bar_range(label_dict): return label_dict.index(BAR_START), label_dict.index(BAR_END)+1
def pos_range(label_dict): return label_dict.index(POS_START), label_dict.index(POS_END)+1
def ins_range(label_dict): return label_dict.index(INS_START), label_dict.index(INS_END)+1
def pitch_range(label_dict): return label_dict.index(PITCH_START), label_dict.index(PITCH_END)+1
def dur_range(label_dict): return label_dict.index(DUR_START), label_dict.index(DUR_END)+1
def vel_range(label_dict): return label_dict.index(VEL_START), label_dict.index(VEL_END)+1
def sig_range(label_dict): return label_dict.index(SIG_START), label_dict.index(SIG_END)+1
def tempo_range(label_dict): return label_dict.index(TEMPO_START), label_dict.index(TEMPO_END)+1
def top_k_top_p(logits_batch, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """

    logits_batch = logits_batch.clone()

    # print(logits_batch.dim())

    if(logits_batch.dim() == 1):
      logits_batch = logits_batch.unsqueeze(0)

    assert logits_batch.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    
    # iterate through batch size 
    for index, logits in enumerate(logits_batch):
      top_k = min(top_k, logits.size(-1))  # Safety check
      if top_k > 0:
          # Remove all tokens with a probability less than the last token of the top-k
          indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
          logits[indices_to_remove] = filter_value

      if top_p > 0.0:
          sorted_logits, sorted_indices = torch.sort(logits, descending=True)
          cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

          # Remove tokens with cumulative probability above the threshold
          sorted_indices_to_remove = cumulative_probs > top_p
          # Shift the indices to the right to keep also the first token above the threshold
          sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
          sorted_indices_to_remove[..., 0] = 0

          indices_to_remove = sorted_indices[sorted_indices_to_remove]
          logits[indices_to_remove] = filter_value
    return logits_batch
# The tokens should be in order (`0-bar`, `1-position`, `2-instrument`, `3-pitch`, `4-duration`, `5-velocity`, `6-timesig` , `7-tempo`) so we switch temperature value accordingly
# Limit to some specific fields such as pitch temp, duration temp, velocity temp, instrument temp  
def switch_temperature(prev_index: int, label_dict, temperature_dict):
  """ Changes temperature to value for one of the eight fields in octuple 
      Args: 
        logits: logits distribution shape (vocabulary size)
        prev_index: previous predicted token 
        label_dict : dictionary mapping string octuple encodings to indices 
        temperature_dict : dict containing temperature values for all the 8 individual octuple elements 
      Returns: next temperature value 
  """
  # First we convert the token to it's string mapping 
  prev_index = prev_index.item()
  rev_inv_map = reverse_label_dict(label_dict)
  str_encoding = rev_inv_map[prev_index]
  
  # print(((int(str_encoding[1]) + 1)%8))
  # print(str_encoding)

  return temperature_dict[((int(str_encoding[1]) + 1)%(8))] 

def filter_invalid_indexes(logits, prev_index, label_dict, filter_value=-float('Inf')):
  """ Filter a distribution of logits using prev_predicted token 
        Args:
            logits: logits distribution shape (vocabulary size)
            prev_index: previous predicted token 
            label_dict : dictionary mapping string octuple encodings to indices 
      Returns: filtered logits according to prev_idx 
  """
  
  logits = logits.clone()
    
  prev_index = prev_index.item()
  rev_inv_map = reverse_label_dict(label_dict)
  str_encoding = rev_inv_map[prev_index]

  # For example if previous index was pitch than according to Octuple encoding next note should be duration 
  # Therefore we fill up all the other 7 element ranges with infinity
  
  for tok in SPECIAL_TOKENS:
      logits[label_dict.index(tok)] = filter_value

  # if previous token was 'bar' then we mask everything excluding 'pos' 
  if(str_encoding[1] == '0'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # pos
  elif(str_encoding[1] == '1'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # ins
  elif(str_encoding[1] == '2'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # pitch
  elif(str_encoding[1] == '3'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # dur
  elif(str_encoding[1] == '4'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # vel
  elif(str_encoding[1] == '5'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value
  # sig
  elif(str_encoding[1] == '6'):
    logits[list(range(*bar_range(label_dict)))] = filter_value
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
  # tempo
  elif(str_encoding[1] == '7'):
    logits[list(range(*pos_range(label_dict)))] = filter_value
    logits[list(range(*ins_range(label_dict)))] = filter_value
    logits[list(range(*pitch_range(label_dict)))] = filter_value
    logits[list(range(*dur_range(label_dict)))] = filter_value
    logits[list(range(*vel_range(label_dict)))] = filter_value
    logits[list(range(*sig_range(label_dict)))] = filter_value
    logits[list(range(*tempo_range(label_dict)))] = filter_value

  return logits 
class PRED_MODE(Enum):
    VANILLA = 1
    OCTUPLE_MODE = 2

#Helper function for OCTUPLE_MODE and MULTI_OCTUPLE_MODE
# https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
def split_multi_oct(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list(chain(*list_a[i:i + chunk_size])) 

#Predict missing masks in sequence from left to right

'''
NOTE: Predicts ONLY the masked octuples provided in `masked_octuples` if `prediction_mode` is NOT Vanilla
Else if prediction_mode is Vanilla, it predicts all the masks in the input `octuplemidi_token_encoding`
'''

#octuplemidi_token_encoding: of the format torch.Tensor([0,0,0,0,0,0,0,0, ..........., 2,2,2,2,2,2,2,2]), where 0 is label_dict.bos_idx & 2 is label_dict.eos_idx
#prediction_mode: decides the speed of the mask prediction
#mask_attribs: decides which of the (`bar`, `position`, `instrument`, `pitch`, `duration`, `velocity`, `timesig` , `tempo`) are masked

#masked_octuples: List of the octuple indices in `octuplemidi_token_encoding` that are masked, note that the element indices for `bar` field of these elements would be (mask_octuple_idxs * 8)

def predict_all_masks(roberta_model, roberta_label_dict, temperature_dict, octuplemidi_token_encoding:torch.Tensor, masked_octuples:list = None,
                      prediction_mode:PRED_MODE = PRED_MODE.VANILLA, mask_attribs:list = [3,4,5] ,num_multi_octuples:int = None,
                      temperature = 1.0, top_k=30, top_p=0.6,
                      verbose = False):
  mask_idx = 1236
  octuplemidi_token_encoding = octuplemidi_token_encoding.clone()

  try:
    assert octuplemidi_token_encoding.dim() == 1
  except:
    raise Exception('Please input single dimensional octuple sequence')

  try:
    bos_idx = roberta_label_dict.bos_index
    eos_idx = roberta_label_dict.eos_index
    tens_type = octuplemidi_token_encoding.dtype
    assert torch.equal(octuplemidi_token_encoding[:8], torch.Tensor([bos_idx]*8).type(tens_type)) and \
      torch.equal(octuplemidi_token_encoding[-8:], torch.Tensor([eos_idx]*8).type(tens_type))
  except:
    print('Start:', octuplemidi_token_encoding[:8] )
    print(torch.Tensor([bos_idx]*8))
    print('End:', octuplemidi_token_encoding[-8:])
    print(torch.Tensor([bos_idx]*8))
    raise Exception('`octuplemidi_token_encoding` either does not have 8 <s> tokens or 8 </s> tokens at beginning and end')


  #---------------------------------------------------------------
  # Altering input mask list based on `prediction_mode`
  #---------------------------------------------------------------

  mask_indices = None

  # If `masked_octuples` not provided, then the `prediction_mode` MUST be Vanilla
  if masked_octuples == None:
    try:  
      assert prediction_mode == PRED_MODE.VANILLA
    except:
      #Since the current faster implementations involves the premise that in all the masked notes, same fields of each octuple is masked,
      #For example, we are not considering that in the sequence one octuple has just `duration` masked and another has just `pitch` masked
      raise Exception("Error: Please choose `prediction_mode` as Vanilla since `masked_octuples` is not provided, to use faster modes provide `mask_indices`")

    mask_indices = [i for i, x in enumerate(octuplemidi_token_encoding.tolist()) if x == mask_idx]

  elif prediction_mode == PRED_MODE.VANILLA:

    print('Warning: Ignoring `masked_octuples`, `mask_attribs` & `num_multi_octuples` as `prediction_mode` is set as Vanilla')
    mask_indices = [i for i, x in enumerate(octuplemidi_token_encoding.tolist()) if x == mask_idx]
    
  elif prediction_mode == PRED_MODE.OCTUPLE_MODE:

    if num_multi_octuples is not None:
      print('Warning: Ignoring `num_multi_octuples` as `prediction_mode` is set as Octuple mode (not Multi-octuple mode)')

    mask_indices = [ [x*8 + y for y in mask_attribs] for x in masked_octuples]

  else:
    raise Exception("Invalid `prediction_mode`")


  try:
    assert len(mask_indices) > 0
  except AssertionError:
    raise Exception('Please input sentence tokens with at least one mask token')

  try:
    assert all( torch.all(octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx) for octuple_midi_mask_elem in mask_indices )
  except:
    print([octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx for octuple_midi_mask_elem in mask_indices])
    raise Exception('Fatal error: At least one element of `mask_indices` is not <mask> (1236)')
  
  #--------------------------------------------------------------
  # Inputting masked indices to model using `prediction_strategy`
  #--------------------------------------------------------------

  if prediction_mode == PRED_MODE.VANILLA:
    pass

  elif prediction_mode == PRED_MODE.OCTUPLE_MODE:
    
    #Checking if `mask_attribs` is fine
    try:
      mask_attribs_len = len(mask_attribs)
      assert mask_attribs_len > 0 and \
      len(set(mask_attribs)) == mask_attribs_len and \
      all( (x >= 0 and x < 8) for x in mask_attribs)
    except:
      raise Exception("`mask_attribs` not appropriate")

  print(f'Final mask indices list sent to model: {mask_indices}')
  print(f'Out of total input size: {len(octuplemidi_token_encoding.tolist())}')

  
  #Let's take an example where `mask_attribs` = [3,4,5] and at least 

  # `final_mask_indices` is of form [3,4,5,11,12,13....] in `Vanilla` prediction mode
  # `final_mask_indices` is of form [[3,4,5],[11,12,13]......] in `Octuple` prediction mode
  # `final_mask_indices` is of form [[3,4,5,11,12,13]......] in `Multi Octuple` prediction mode, in this case num_multi_octuples = 2

  filter_value = -float('Inf')
  octuplemidi_token_encoding_device = octuplemidi_token_encoding.device

  # Finally predicting masks based on `prediction_strategy`

  ################################ 
  # Vanilla Mode Prediction mode #
  ################################

  if prediction_mode == PRED_MODE.VANILLA: 
    
    repeat_count = 0
    
    for mask_idx_batch in tqdm(mask_indices):
        input = octuplemidi_token_encoding.unsqueeze(0).cuda()
        prev_idx = octuplemidi_token_encoding[mask_idx_batch-1]

        with torch.no_grad():
          # extr_features shape -> [1, 8016, 1237] 
          # Here 1 is the batch size, 8016 is embedding dimension and 1237 is the vocab size 
          extr_features, _ = roberta_model.model.extract_features(input)
          # filter the mask indices from the extracted feature tokens (1000 octuples) 
          logits = extr_features[0, mask_idx_batch]

          temperature = switch_temperature(prev_idx, roberta_label_dict, temperature_dict)
          repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
          temperature += repeat_penalty

          if temperature != 1. : logits = logits/temperature

          logits = filter_invalid_indexes(logits, prev_idx, roberta_base.task.label_dictionary, filter_value=filter_value)
          logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)

          probs = torch.softmax(logits, dim = -1)
          if probs.dim() == 1:
            probs = probs.unsqueeze(0)

          # Update repeat count
          num_choices = len(probs.nonzero().reshape(-1))
          if num_choices <= 2: repeat_count += 1
          else: repeat_count = repeat_count // 2

          if(temperature != 1 or top_k != 0 or top_p != 0):
            # We sample from a multinomial distribution
            top_preds = torch.multinomial(probs, 1)
            top_preds = top_preds.reshape(-1)
          else:
            # We take the argmax or only choose the top candidate
            # print('Predicting argmax mode!')
            top_preds = torch.argmax(probs, dim = 1)
        
        # Assign the token 
        octuplemidi_token_encoding_type = octuplemidi_token_encoding.dtype
        octuplemidi_token_encoding[mask_idx_batch] = top_preds.\
                                                    type(octuplemidi_token_encoding_type). \
                                                    to(octuplemidi_token_encoding_device)
 
  ################################ 
  # Octuple Mode Prediction mode #
  ################################

  # For Octuple mod switching temperature is not possible so we only use one value 
  elif prediction_mode == PRED_MODE.OCTUPLE_MODE: 
    
    # iterate through all the mask indices 
    for mask_idx_batch in tqdm(mask_indices):
        input = octuplemidi_token_encoding.unsqueeze(0).cuda()
        
        with torch.no_grad():
          # extr_features shape -> [1, 8016, 1237] 
          # Here 1 is the batch size, 8016 is embedding dimension and 1237 is the vocab size 
          extr_features, _ = roberta_model.model.extract_features(input)
          # filter the mask indices from the extracted feature tokens (1000 octuples) 
          logits = extr_features[0, mask_idx_batch]

          # Apply Temperature if not equal to 1 
          if temperature != 1. : logits = logits/temperature
          
          # Apply top-k and top-p if != 0 
          logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)

          probs = torch.softmax(logits, dim = -1)
          if probs.dim() == 1:
            probs = probs.unsqueeze(0)

          if(temperature != 1 or top_k != 0 or top_p != 0):
            # We sample from a multinomial distribution
            top_preds = torch.multinomial(probs, 1)
            top_preds = top_preds.reshape(-1)
          else:
            # We take the argmax or only choose the top candidate
            top_preds = torch.argmax(probs, dim = 1)
        
        octuplemidi_token_encoding_type = octuplemidi_token_encoding.dtype
        octuplemidi_token_encoding[mask_idx_batch] = top_preds.\
                                                    type(octuplemidi_token_encoding_type). \
                                                    to(octuplemidi_token_encoding_device)      

  # Final error check 
  try:
    assert not any( torch.all(octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx) for octuple_midi_mask_elem in mask_indices )
  except:
    print([octuplemidi_token_encoding[octuple_midi_mask_elem] == mask_idx for octuple_midi_mask_elem in mask_indices])
    raise Exception('Fatal error: The prediction has at least one element of `mask_indices` as <mask>')

  return octuplemidi_token_encoding

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

            # 初始化 token_embeddings 為 None，以便後續檢查
            token_embeddings = None
            last_hidden_state_from_model = None # 用於儲存提取出的主要特徵張量

            if isinstance(model_output_result, dict) and 'x' in model_output_result:
                last_hidden_state_from_model = model_output_result['x'] # Shape: [seq_len, batch_size, hidden_size]
                print(f"Extracted from dict['x'], shape: {last_hidden_state_from_model.shape}")
            elif isinstance(model_output_result, tuple) and len(model_output_result) > 0:
                print(f"Model output is a tuple. Length: {len(model_output_result)}")
                # 通常元組的第一個元素是最後的隱藏狀態
                # 我們需要確認它的維度和形狀
                if isinstance(model_output_result[0], torch.Tensor):
                    last_hidden_state_from_model = model_output_result[0]
                    print(f"Extracted from tuple[0], shape: {last_hidden_state_from_model.shape}")
                    # 為了更穩健，可以遍歷元組中的張量，找到維度最匹配的那個
                    # for i, item in enumerate(model_output_result):
                    #     if isinstance(item, torch.Tensor) and item.dim() == 3:
                    #         print(f"Tuple element {i} shape: {item.shape}")
                    #         # 這裡可以加入更多邏輯來選擇正確的張量，例如基於維度或與768的匹配
                    #         if item.shape[-1] == 768: # 假設 hidden_size 是 768
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

            # --- 處理提取出的 last_hidden_state_from_model ---
            if last_hidden_state_from_model is None:
                print("Error: last_hidden_state_from_model is None after attempting extraction.")
                return None

            # 預期 hidden_size 為 768
            expected_hidden_size = 768
            if last_hidden_state_from_model.shape[-1] != expected_hidden_size:
                print(f"Error: Extracted features have incorrect hidden dimension: {last_hidden_state_from_model.shape[-1]}. Expected {expected_hidden_size}.")
                return None

            # 轉換為 [batch_size, seq_len, hidden_size]
            # Fairseq encoder 通常輸出 [seq_len, batch_size, hidden_size]
            if last_hidden_state_from_model.shape[0] == input_tensor_for_model.shape[1] and \
               last_hidden_state_from_model.shape[1] == input_tensor_for_model.shape[0]: # [seq_len, batch (1), hidden_size]
                token_embeddings = last_hidden_state_from_model.transpose(0, 1)
            # 有些實現可能直接輸出 [batch_size, seq_len, hidden_size]
            elif last_hidden_state_from_model.shape[0] == input_tensor_for_model.shape[0] and \
                 last_hidden_state_from_model.shape[1] == input_tensor_for_model.shape[1]: # [batch (1), seq_len, hidden_size]
                token_embeddings = last_hidden_state_from_model
            else:
                print(f"Error: Shape mismatch after feature extraction. Features shape: {last_hidden_state_from_model.shape}, Input shape for model: {input_tensor_for_model.shape}. Cannot determine correct transpose.")
                return None
            
            print(f"Shape of token_embeddings after processing and transpose (if any): {token_embeddings.shape}") # Expected: [1, seq_len, 768]


            # --- 池化 (Pooling) ---
            if pooling_strategy == 'cls':
                sequence_embedding = token_embeddings[:, 0, :]
            elif pooling_strategy == 'mean':
                pad_idx = roberta_label_dict.pad_index if hasattr(roberta_label_dict, 'pad_index') else roberta_label_dict.pad()
                attention_mask_for_pooling = (input_tensor_for_model != pad_idx).to(device) # [batch_size, sequence_length]
                
                # 確保 attention_mask_for_pooling 的序列長度與 token_embeddings 的序列長度匹配
                if attention_mask_for_pooling.shape[1] != token_embeddings.shape[1]:
                    print(f"Warning: Sequence length mismatch between attention mask ({attention_mask_for_pooling.shape[1]}) and token_embeddings ({token_embeddings.shape[1]}). This might lead to incorrect pooling.")
                    # 你可能需要基於 token_embeddings.shape[1] 重新調整 attention_mask_for_pooling 或進行截斷/填充
                    # 為了簡單起見，這裡暫不處理，但實際應用中需要注意

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
        print(f"MIDI parsing failed for {midi_path}.") # 添加一些日誌
        return None # 修改：如果解析失敗，返回 None
    # print(f"> Parsed MIDI: {midi_name} ({len(midi_obj.instruments)} tracks)") # 可以保留或移除

    octuple_encoding_list = MIDI_to_encoding(midi_obj)
    if not octuple_encoding_list:
        print(f"MIDI to encoding failed or resulted in empty sequence for {midi_path}.") # 添加日誌
        return None # 修改：如果編碼失敗，返回 None
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
        print(f"Encoding to string failed for {midi_path}.") # 添加日誌
        return None # 修改：如果轉換失敗，返回 None

    # d. Tokenize the string using MusicBERT's dictionary
    label_dict = roberta_base.task.label_dictionary
    octuple_midi_tokenized = label_dict.encode_line(
        octuple_midi_str,
        append_eos=False, # encoding_to_str 應該已經添加了 BOS/EOS (確認一下)
        add_if_not_exist=False
    )

    # e. Extract embedding
    midi_embedding = extract_midi_embedding(
        roberta_base,
        label_dict,
        octuple_midi_tokenized, # 傳遞 1D tokenized 張量
        pooling_strategy='mean', # 或 'cls'
        device=device # 全局 device
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
        import traceback
        traceback.print_exc()
        return None

class FuseModel(torch.nn.Module):
    def __init__(self, input_dim = 2005, output_dim= 1024, hidden_dim=2048):
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
fuse_model = FuseModel(input_dim=1536, output_dim=512, hidden_dim=2048).to(device)
optimizer = torch.optim.Adam(fuse_model.parameters(), lr=1e-4)
num_epoches = 100
def main():
    proj_to_clip = torch.nn.Linear(1024, 512).to(device).to(dtype=torch.float16)

    datasetPath = './kaggleData/small_song_lyrics.csv'
    # Read the dataset as a CSV file
    dataset = pd.read_csv(datasetPath)
    print(dataset)
    # Iterate through all rows in the dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    os.makedirs(f"./generated_images/{timestamp}", exist_ok=True)
    # os.makedirs(f"./generated_images/{timestamp}/training", exist_ok=True)
    # Freeze Stable Diffusion pipeline submodules (text_encoder and unet)
    for param in clip_model.parameters():
        param.requires_grad = False
    sims = []
    simse = []
    for epoch in range(num_epoches):
        nowS = []
        # if (epoch+1) == 1 or (epoch+1) % 10 == 0:
        #     os.makedirs(f"./generated_images/{timestamp}/training/{epoch}", exist_ok=True)
        for index, row in dataset.iterrows():
            artist = row['Artist']
            title = row['Title']
            title = title.replace('/', '_')
            title = title.replace('.', '_')
            artist = artist.replace('/', '_')
            artist = artist.replace('.', '_')
            lyric = row['Lyric']        # Assuming the dataset has a column named 'lyrics'
            midi_path = f'./trainingData/midi/{artist}-{title}.mid'  # Assuming the dataset has a column named 'midi_path'
            image_path = f'./trainingData/thumbnails/{artist}-{title}.jpg'
            if not os.path.exists(midi_path) or not os.path.exists(image_path):
                print(f'"{midi_path}" or "{image_path}" does not exist, skipping...')
                continue
            print(f"Epoch {epoch}: {artist} {title} {midi_path}")
            if not isinstance(lyric, str) or not lyric.strip():
                print(f'Lyric for "{midi_path}" is empty, skipping...')
                continue
            if not os.path.exists(midi_path):
                print(f'"{midi_path}" not exist')
                continue
            midi_embeding = midiEmbeding(midi_path)
            if midi_embeding == None:
                print(f'"{midi_path}" error')
                continue
            text_embeddings = get_text_embedding(lyric, bert_tokenizer, bert_model, text_bert_device)
            if midi_embeding.dim() == 1: midi_embeding = midi_embeding.unsqueeze(0)
            if text_embeddings.dim() == 1: text_embeddings = text_embeddings.unsqueeze(0)
            midi_embeding = midi_embeding.to(device)
            text_embeddings = text_embeddings.to(device)
            fused_embedding = fuse_model(midi_embeding, text_embeddings)
            sequence_length = 77 # Standard for CLIP-based SD models
            prompt_embeds_final = fused_embedding.unsqueeze(1)
            prompt_embeds_final = prompt_embeds_final.repeat(1, sequence_length, 1)
            prompt_embeds_final = prompt_embeds_final.to('cuda:0', dtype=torch.float32)
            # print(text_embeddings)
            # print(midi_embeding)
            # print(fused_embedding)
            # print(prompt_embeds_final)
            # print(prompt_embeds_final.dtype)
            # prompt_embeds_final_512 = proj_to_clip(prompt_embeds_final)

            # Extract image embedding using Stable Diffusion's CLIP model
            # Use CLIP model to extract image embedding
            answer_image = Image.open(image_path).convert("RGB")
            # Resize the image to 512x512 before processing with CLIPW
            # answer_image = answer_image.resize((512, 512), Image.BICUBIC)
            image = clip_processor(images=answer_image, return_tensors="pt").to(device)
            image_embedding = clip_model.get_image_features(**image)
            print(f"Image embedding extracted with shape: {image_embedding.shape}")
            # Calculate similarity between image embedding and text embedding
            if image_embedding.dim() == 2:  # Ensure image embedding is 2D
                image_embedding = image_embedding.unsqueeze(1).repeat(1, prompt_embeds_final.shape[1], 1) # Pool across sequence length
            # Normalize embeddings
            image_embedding = torch.nn.functional.normalize(image_embedding, p=2, dim=-1)
            prompt_embeds_final = torch.nn.functional.normalize(prompt_embeds_final, p=2, dim=-1)

            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(image_embedding, prompt_embeds_final, dim=-1)
            print(f"Cosine similarity between image and text: {similarity.mean().item()}")

            # Use similarity as loss (maximize similarity, so minimize -similarity)
            loss = 1-similarity.mean()
            sims.append(similarity.mean().detach().cpu().numpy())
            nowS.append(similarity.mean().detach().cpu().numpy()) 
            # loss = loss * 1000
            print(f"Loss: {loss.item()}")

            # Backward pass
            fuse_model.zero_grad()
            loss.backward()
            for name, param in fuse_model.named_parameters():
                if param.grad is None:
                    print(f"{name} has no gradient")
            optimizer.step()
            # if (epoch+1) == 1 or (epoch+1) % 10 == 0:
            #     output_image_path = f"./generated_images/{timestamp}/training/{epoch}/{artist}-{title}.png"
            #     generated_image = sd_pipe(prompt_embeds=prompt_embeds_final).images[0]
            #     generated_image.save(output_image_path)
            #     print(f"Image generated and saved to {output_image_path}")
        if len(nowS) > 0:
            mean_nowS = sum(nowS) / len(nowS)
            print(f"Epoch {epoch}: Mean similarity for this epoch: {mean_nowS}")
            simse.append(mean_nowS)
        else:
            print(f"Epoch {epoch}: No similarity values computed.")
    # Save the trained model
    model_save_path = f"./generated_images/{timestamp}/fuse_model.pth"
    torch.save(fuse_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    import matplotlib.pyplot as plt

    # Plot the similarity values over training steps
    plt.figure(figsize=(10, 5))
    plt.plot(sims, label='Cosine Similarity')
    plt.xlabel('Training Step')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between Image and Fused Embedding during Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./generated_images/{timestamp}/similarity_plot.png")
    # Plot the mean similarity per epoch (simse)
    plt.figure(figsize=(10, 5))
    plt.plot(simse, label='Mean Similarity per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Cosine Similarity')
    plt.title('Mean Cosine Similarity per Epoch during Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./generated_images/{timestamp}/mean_similarity_per_epoch.png")
    embed_save_dir = f'./generated_images/{timestamp}/embeddings/'
    
    os.makedirs(embed_save_dir, exist_ok=True)
    for index, row in dataset.iterrows():
        artist = row['Artist']
        title = row['Title']
        title = title.replace('/', '_')
        title = title.replace('.', '_')
        artist = artist.replace('/', '_')
        artist = artist.replace('.', '_')
        midi_path = f'./trainingData/midi/{artist}-{title}.mid'  # Assuming the dataset has a column named 'midi_path'
        lyric = row['Lyric']        # Assuming the dataset has a column named 'lyrics'
        if not isinstance(lyric, str) or not lyric.strip():
            print(f'Lyric for "{midi_path}" is empty, skipping...')
            continue
        if not os.path.exists(midi_path):
            print(f'"{midi_path}" not exist')
            continue
        midi_embeding = midiEmbeding(midi_path)
        if midi_embeding == None:
            print('"{midi_path}" error')
            continue
        text_embeddings = get_text_embedding(lyric, bert_tokenizer, bert_model, text_bert_device)
        if midi_embeding.dim() == 1: midi_embeding = midi_embeding.unsqueeze(0)
        if text_embeddings.dim() == 1: text_embeddings = text_embeddings.unsqueeze(0)
        midi_embeding = midi_embeding.to(device)
        text_embeddings = text_embeddings.to(device)
        fused_embedding = fuse_model(midi_embeding, text_embeddings)
        sequence_length = 77 # Standard for CLIP-based SD models
        prompt_embeds_final = fused_embedding.unsqueeze(1)
        prompt_embeds_final = prompt_embeds_final.repeat(1, sequence_length, 1)
        prompt_embeds_final = prompt_embeds_final.to('cuda:0', dtype=torch.float32)
        embed_save_path = os.path.join(embed_save_dir, f'{artist}-{title}.pt')
        print(prompt_embeds_final)
        torch.save(prompt_embeds_final.cpu(), embed_save_path)
        
    #     with torch.no_grad():
    #         # Pass the correctly shaped embedding to prompt_embeds
    #         generated_image = sd_pipe(prompt_embeds=prompt_embeds_final).images[0]

    #     # Save the generated image
    #     output_image_path = f"./generated_images/{timestamp}/{artist}-{title}.png"
    #     generated_image.save(output_image_path)
    #     print(f"Image generated and saved to {output_image_path}")
    return 0
    midi_path = './trainingData/midi/Rick Astley-Never Gonna Give You Up.mid'
    lyric = "We're no strangers to love You know the rules and so do I (do I) A full commitment's what I'm thinking of You wouldn't get this from any other guy I just wanna tell you how I'm feeling Gotta make you understand Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you We've known each other for so long Your heart's been aching  but you're too shy to say it (say it) Inside  we both know what's been going on (going on) We know the game and we're gonna play it And if you ask me how I'm feeling Don't tell me you're too blind to see Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you We've known each other for so long Your heart's been aching  but you're too shy to say it (to say it) Inside  we both know what's been going on (going on) We know the game and we're gonna play it I just wanna tell you how I'm feeling Gotta make you understand Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you"
    midi_embeding = midiEmbeding(midi_path)
    print(midi_embeding)
    text_embeddings = get_text_embedding(lyric, bert_tokenizer, bert_model, text_bert_device)
    print(text_embeddings)
    # Fuse MIDI and text embeddings
    # Ensure both embeddings are 2D: [1, hidden_size] and on the same device
    if midi_embeding.dim() == 1: midi_embeding = midi_embeding.unsqueeze(0)
    if text_embeddings.dim() == 1: text_embeddings = text_embeddings.unsqueeze(0)
    midi_embeding = midi_embeding.to(device) # Use the main device
    text_embeddings = text_embeddings.to(device) # Use the main device

    # Concatenate embeddings along the feature dimension
    fused_embedding = fuse_model(midi_embeding, text_embeddings)
    

    if fused_embedding is not None:
        # Use a linear layer to adjust the fused embedding to the required size
        # For stable-diffusion-2-1-base, the expected dim is 1024
        try:
            # --- Reshape for Stable Diffusion ---
            # SD expects [batch_size, sequence_length, embedding_dim]
            # We need to make our [1, 1024] look like [1, 77, 1024]
            # Add the sequence length dimension and repeat
            sequence_length = 77 # Standard for CLIP-based SD models
            # Add sequence dimension: [1, 1, 1024]
            prompt_embeds_final = fused_embedding.unsqueeze(1)
            # Repeat along the sequence dimension: [1, 77, 1024]
            prompt_embeds_final = prompt_embeds_final.repeat(1, sequence_length, 1)

            print(f"Final prompt_embeds shape for SD: {prompt_embeds_final.shape}") # Should be [1, 77, 1024]

            # Ensure it's on the same device as the pipeline and correct dtype
            prompt_embeds_final = prompt_embeds_final.to(sd_pipe.device, dtype=sd_pipe.text_encoder.dtype)


            # --- Generate Image ---
            print("Generating image using fused embeddings...")
            with torch.no_grad():
                # Pass the correctly shaped embedding to prompt_embeds
                generated_image = sd_pipe(prompt_embeds=prompt_embeds_final).images[0]

            # Save the generated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_image_path = f"./generated_image_{timestamp}.png"
            generated_image.save(output_image_path)
            print(f"Image generated and saved to {output_image_path}")

        except Exception as e:
            print(f"Error during linear projection or image generation: {e}")
            import traceback
            traceback.print_exc()
    # else:
    #     print("Error: One or both embeddings are None. Cannot fuse embeddings.")
if __name__ == "__main__":
    main()