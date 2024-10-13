import spaces
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'amt/src')))

import subprocess
from typing import Tuple, Dict, Literal
from ctypes import ArgumentError

from model_helper import *

import torchaudio
import glob
import gradio as gr
from gradio_log import Log
from pathlib import Path

# gradio_log
log_file = 'amt/log.txt'
Path(log_file).touch()
# @title Load Checkpoint
model_name = 'YPTF.MoE+Multi (noPS)' # @param ["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]
precision = '16'# if torch.cuda.is_available() else '32'# @param ["32", "bf16-mixed", "16"]
project = '2024'

if model_name == "YMT3+":
    checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
    args = [checkpoint, '-p', project, '-pr', precision]
elif model_name == "YPTF+Single (noPS)":
    checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
    args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
            '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF+Multi (PS)":
    checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256',
            '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
            '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF.MoE+Multi (noPS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF.MoE+Multi (PS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
else:
    raise ValueError(model_name)
model = load_model_checkpoint(args=args, device="cpu")
model.to("cuda")
def prepare_media(source_path_or_url: os.PathLike,
                  source_type: Literal['audio_filepath', 'youtube_url'],
                  delete_video: bool = True,
                  simulate = False) -> Dict:
    """prepare media from source path or youtube, and return audio info"""
    # Get audio_file
    if source_type == 'audio_filepath':
        audio_file = source_path_or_url
    elif source_type == 'youtube_url':
        if os.path.exists('/download/yt_audio.mp3'):
            os.remove('/download/yt_audio.mp3')
        # Download from youtube
        with open(log_file, 'w') as lf:
            audio_file = './downloaded/yt_audio'
            command = ['yt-dlp', '-x', source_path_or_url, '-f', 'bestaudio',
                '-o', audio_file, '--audio-format', 'mp3', '--restrict-filenames',
                '--extractor-retries', '10',
                '--force-overwrites', '--username', 'oauth2', '--password', '', '-v']
            if simulate:
                command = command + ['-s']
            process = subprocess.Popen(command,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
            for line in iter(process.stdout.readline, ''):
                # Filter out unnecessary messages
                print(line)
                if "www.google.com/device" in line:
                    hl_text = line.replace("https://www.google.com/device", "\033[93mhttps://www.google.com/device\x1b[0m").split()
                    hl_text[-1] = "\x1b[31;1m" + hl_text[-1] + "\x1b[0m"
                    lf.write(' '.join(hl_text)); lf.flush()
                elif "Authorization successful" in line or "Video unavailable" in line:
                    lf.write(line); lf.flush()
            process.stdout.close()
            process.wait()
        
        audio_file += '.mp3'
    else:
        raise ValueError(source_type)

    # Create info
    info = torchaudio.info(audio_file)
    return {
        "filepath": audio_file,
        "track_name": os.path.basename(audio_file).split('.')[0],
        "sample_rate": int(info.sample_rate),
        "bits_per_sample": int(info.bits_per_sample),
        "num_channels": int(info.num_channels),
        "num_frames": int(info.num_frames),
        "duration": int(info.num_frames / info.sample_rate),
        "encoding": str.lower(info.encoding),
        }
@spaces.GPU
def process_audio(audio_filepath, midi_filepath):
    if audio_filepath is None:
        return None
    audio_info = prepare_media(audio_filepath, source_type='audio_filepath')
    midifile = transcribe(model, audio_info)
    with open(midi_filepath, 'wb') as f:
        f.write(midifile)
audio = input('audiopath')
midi = input('midipath')
process_audio(audio, midi)