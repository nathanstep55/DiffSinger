# coding=utf8
import argparse
import os
import pathlib
import sys

root_dir = pathlib.Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import tqdm
import librosa
import onnx
import pyworld as pw

from utils.binarizer_utils import get_pitch_parselmouth, get_mel_torch
from modules.vocoders.ddsp import DDSP
from modules.vocoders.nsf_hifigan import NsfHifiGAN
from utils.infer_utils import cross_fade, save_wav
from utils.hparams import set_hparams, hparams
from modules.pe.rmvpe import RMVPE

parser = argparse.ArgumentParser(description='Use a vocoder to transpose the pitch of vocals')
parser.add_argument('audfile', type=str, help='Path to the input file (.wav, .mp3, .ogg, .flac, etc.)')
parser.add_argument('--transpose', type=float, required=False, help='Amount of semitones to transpose the pitch of the input')
parser.add_argument('--method', type=str, required=False, help='Specify vocoding method ("world", "nsfhifigan", "ddsp")')
parser.add_argument('--exp', type=str, required=False, help='Read vocoder class and path from chosen experiment')
parser.add_argument('--config', type=str, required=False, help='Read vocoder class and path from config file')
parser.add_argument('--class', type=str, required=False, help='Specify vocoder class')
parser.add_argument('--ckpt', type=str, required=False, help='Specify vocoder checkpoint path')
parser.add_argument('--onnx', type=str, required=False, help='Specify vocoder model onnx')
parser.add_argument('--pitchalgo', type=str, required=False, help='Specify pitch algorithm ("parselmouth" or "rmvpe")')
parser.add_argument('--rmvpeckpt', type=str, required=False, help='Specify RMVPE checkpoint path')
parser.add_argument('--out', type=str, required=False, help='Path of the output folder')
parser.add_argument('--title', type=str, required=False, help='Title of output file')
args = parser.parse_args()

method = 'nsfhifigan'
if args.method:
    method = args.method

# transposing factor in semitones
transpose_semitones = args.transpose
if not args.transpose:
    transpose_semitones = 0.0

audfile = pathlib.Path(args.audfile)
name = f"{audfile.stem}-{method}-{transpose_semitones}-out" if not args.title else args.title
config = None
if method != 'world':
    if args.exp:
        config = root_dir / 'checkpoints' / args.exp / 'config.yaml'
    elif args.config:
        config = pathlib.Path(args.config)
    else:
        assert False, 'Either argument \'--exp\' or \'--config\' should be specified.'

    sys.argv = [
        sys.argv[0],
        '--config',
        str(config)
    ]
    set_hparams(print_hparams=False)

cls = getattr(args, 'class')
if cls:
    hparams['vocoder'] = cls
vocoder = None

if args.onnx:
    vocoder = onnx.load(args.onnx)
elif args.ckpt:
    hparams['vocoder_ckpt'] = args.ckpt

# get sample rate from model config
sample_rate = 44100
if 'audio_sample_rate' in hparams:
    sample_rate = hparams.get('audio_sample_rate')
elif 'sampling_rate' in hparams:  # openvpi
    sample_rate = hparams.get('sampling_rate')
 
# get mel bins
num_mel_bins = None
if 'audio_num_mel_bins' in hparams:
    num_mel_bins = hparams.get('audio_num_mel_bins')
elif 'num_mels' in hparams:  # openvpi
    num_mel_bins = hparams.get('num_mels')

# hop size
hop_size = None
if 'hop_size' in hparams:
    hop_size = hparams.get('hop_size')

# window size
win_size = None
if 'win_size' in hparams:
    win_size = hparams.get('win_size')

# fft size
fft_size = None
if 'fft_size' in hparams:
    fft_size = hparams.get('fft_size')
elif 'n_fft' in hparams:  # openvpi
    fft_size = hparams.get('n_fft')

# minimum frequency
fmin = None
if 'fmin' in hparams:
    fmin = hparams.get('fmin')

# maximum frequency
fmax = None
if 'fmax' in hparams:
    fmax = hparams.get('fmax')

rmvpe = None
if args.pitchalgo == 'rmvpe':
    if not args.rmvpeckpt:
        print("Warning: RMVPE checkpoint path not specified, using parselmouth for F0 detection instead")
    else:
        rmvpe = RMVPE(args.rmvpeckpt)

out = args.out
if args.out:
    out = pathlib.Path(args.out)
else:
    out = audfile.parent

# Load audio using librosa
# Internally this should use libsndfile or audioread, so a lot of formats are supported
aud, _ = librosa.load(audfile, sr=sample_rate, mono=True, dtype=np.float32 if method != 'world' else np.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if vocoder is None:
    if method == 'nsfhifigan':
        vocoder = NsfHifiGAN()
    elif method == 'ddsp':
        vocoder = DDSP()

def run_vocoder(path: pathlib.Path):
    
    if method == 'world':
        
        _f0, t = pw.dio(aud, sample_rate)    # raw pitch extractor
        f0 = pw.stonemask(aud, _f0, t, sample_rate)  # pitch refinement
        sp = pw.cheaptrick(aud, f0, t, sample_rate)  # extract smoothed spectrogram
        ap = pw.d4c(aud, f0, t, sample_rate)         # extract aperiodicity

        # Shift pitch by some semitones
        f0 *= 2.0 ** (transpose_semitones / 12.0)

        wav_out = pw.synthesize(f0, sp, ap, sample_rate) # synthesize an utterance using the parameters

    else:

        # Generate mel spectrogram from audio file
        mel = get_mel_torch(
            aud, sample_rate, num_mel_bins=num_mel_bins,
            hop_size=hop_size, win_size=win_size, fft_size=fft_size,
            fmin=fmin, fmax=fmax,
            device=device
        )

        if rmvpe is not None:
            f0, _ = rmvpe.get_pitch(
                aud, samplerate=sample_rate, length=len(mel),
                hop_size=hop_size
            )
        else:
            f0, _ = get_pitch_parselmouth(
                aud, samplerate=sample_rate, length=len(mel),
                hop_size=hop_size
            )

        # Shift pitch by some semitones
        f0 *= 2.0 ** (transpose_semitones / 12.0)

        y = vocoder.spec2wav_torch(mel, f0=f0)

        wav_out = y.cpu().numpy()

    print(f'| save audio: {path}')
    save_wav(wav=wav_out, path=path, sr=sample_rate)


os.makedirs(out, exist_ok=True)
try:
    run_vocoder(out / (name + '.wav'))
except KeyboardInterrupt:
    exit(-1)
