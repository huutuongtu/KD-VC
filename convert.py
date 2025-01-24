import os
import argparse
import torch
import librosa
import time
import json
from scipy.io.wavfile import write
from tqdm import tqdm
from utils import latest_checkpoint_path
import utils
from models_kd import SynthesizerTrn as kd_model
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from resemblyzer import VoiceEncoder, preprocess_wav
import logging
from pathlib import Path
logging.getLogger('numba').setLevel(logging.WARNING)


if __name__ == "__main__":

    src = "src.wav"
    tgt = "tgt.wav"
    out_audio = "out1.wav"

    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, help="path to json config file")
    parser.add_argument("--ptfile", type=str, help="path to pth file")
    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = kd_model(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("loading speaker encoder resembly...")
        encoder = VoiceEncoder()

    print("Synthesizing...")
    with torch.no_grad():
        wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        if hps.model.use_spk:
            g_tgt = encoder.embed_utterance(preprocess_wav(wav_tgt))
            g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
        else:
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            mel_tgt = mel_spectrogram_torch(
                wav_tgt, 
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
        # src
        wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
        c = utils.get_content(cmodel, wav_src)
        
        if hps.model.use_spk:
            audio = net_g.infer(c, g=g_tgt)
        else:
            audio = net_g.infer(c, mel=mel_tgt)
        audio = audio[0][0].data.cpu().float().numpy()
        write(out_audio, hps.data.sampling_rate, audio)
