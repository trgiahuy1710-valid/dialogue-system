import os
import shutil
import yaml
import torch
import librosa
import numpy as np
import soundfile as sf
import torchaudio
import re
import json
from glob import glob
from munch import Munch
from nltk.tokenize import word_tokenize, sent_tokenize

# --- 1. CẤU HÌNH HỆ THỐNG (Giữ nguyên từ script cũ của bạn) ---
os.environ['PHONEMIZER_ESPEAK_EXECUTABLE'] = r'C:\Program Files\eSpeak NG\espeak-ng.exe'
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'

import phonemizer
from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# =============================
# SETUP STYLE TTS 2 COMPONENTS
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
global_phonemizer = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
textcleaner = TextCleaner()
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch({k: recursive_munch(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1)
    mask = mask.type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000: audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def load_models():
    global model, sampler, model_params
    # Lưu ý: Kiểm tra lại các đường dẫn này có đúng trên máy bạn không
    libri_tts_path = r"E:\Python_Project\dialouge test\StyleTTS2\Models\LibriTTS\config.yml"
    config = yaml.safe_load(open(libri_tts_path))

    text_aligner = load_ASR_models(config.get("ASR_path"), config.get("ASR_config"))
    pitch_extractor = load_F0_models(config.get("F0_path"))
    plbert = load_plbert(config.get("PLBERT_dir"))

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    model_path = r"E:\Python_Project\dialouge test\StyleTTS2\Models\LibriTTS\epochs_2nd_00020.pth"
    params = torch.load(model_path, map_location="cpu")["net"]

    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items(): new_state_dict[k[7:]] = v
                model[key].load_state_dict(new_state_dict, strict=False)
        model[key].eval().to(device)

    sampler = DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(),
                               sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)


def inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    tokens = textcleaner(" ".join(word_tokenize(ps[0])))
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device), embedding=bert_dur,
                         embedding_scale=embedding_scale, features=ref_s, num_steps=diffusion_steps).squeeze(1)

        s = beta * s_pred[:, 128:] + (1 - beta) * ref_s[:, 128:]
        ref = alpha * s_pred[:, :128] + (1 - alpha) * ref_s[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln.size(0)):
            pred_aln[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln.unsqueeze(0).to(device)), F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50]


def crossfade(a, b, fade_samples=1200):
    fade_samples = min(fade_samples, len(a), len(b))
    if fade_samples <= 0: return np.concatenate([a, b])
    t = np.linspace(0, np.pi, fade_samples)
    fade_out, fade_in = (1 + np.cos(t)) / 2, (1 - np.cos(t)) / 2
    merged = a[-fade_samples:] * fade_out + b[:fade_samples] * fade_in
    return np.concatenate([a[:-fade_samples], merged, b[fade_samples:]])


def inference_long(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1):
    chunks = sent_tokenize(text)
    audio = None
    for chunk in chunks:
        if len(chunk) < 2: continue
        try:
            wav = inference(chunk, ref_s, alpha, beta, diffusion_steps, embedding_scale)
        except:
            continue
        wav = np.asarray(wav).flatten()
        audio = wav if audio is None else crossfade(audio, wav)
    return audio if audio is not None else np.zeros(2400, dtype=np.float32)


# =============================
# MOSHI DATA GENERATION LOGIC
# =============================

def create_stereo_audio(left_wav, right_wav):
    max_len = max(len(left_wav), len(right_wav))
    l_p = np.pad(left_wav, (0, max_len - len(left_wav)), 'constant')
    r_p = np.pad(right_wav, (0, max_len - len(right_wav)), 'constant')
    return np.stack([l_p, r_p], axis=1)  # Left=Moshi, Right=User


def generate_mock_alignment(text, start_time, duration, speaker_id):
    words = text.split()
    if not words: return []
    w_dur = duration / len(words)
    return [[w, [round(start_time + i * w_dur, 2), round(start_time + (i + 1) * w_dur, 2)], speaker_id] for i, w in
            enumerate(words)]


def process_moshi_dataset(input_folder, output_root):
    load_models()
    audio_out = os.path.join(output_root, "audio")
    os.makedirs(audio_out, exist_ok=True)

    # Ref Audio (Cần tồn tại file này)
    ref_u = compute_style(r"E:\Python_Project\dialouge test\StyleTTS2\Demo\reference_audio\Gavin.wav")
    ref_a = compute_style(r"E:\Python_Project\dialouge test\StyleTTS2\Demo\reference_audio\1221-135767-0014.wav")

    json_files = glob(os.path.join(input_folder, "conv_claim_*.json"))
    manifest = []

    for j_path in json_files:
        f_id = os.path.basename(j_path).replace(".json", "")
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        f_l, f_r, aligns = np.array([], dtype=np.float32), np.array([], dtype=np.float32), []
        curr_offset = 0

        for msg in data["messages"]:
            text = msg["content"]
            if msg["role"] == "assistant":
                wav = inference_long(text, ref_a, alpha=0.7, beta=0.3)
                tag = "SPEAKER_01"  # Moshi (Left channel)
                f_l = np.concatenate([f_l, wav])
                f_r = np.concatenate([f_r, np.zeros_like(wav)])
            else:
                wav = inference_long(text, ref_u, alpha=0.2, beta=0.8)
                tag = "SPEAKER_00"  # User (Right channel)
                f_r = np.concatenate([f_r, wav])
                f_l = np.concatenate([f_l, np.zeros_like(wav)])

            dur = len(wav) / 24000
            aligns.extend(generate_mock_alignment(text, curr_offset / 24000, dur, tag))
            curr_offset += len(wav)

        st_wav = create_stereo_audio(f_l, f_r)
        sf.write(os.path.join(audio_out, f"{f_id}.wav"), st_wav, 24000)
        with open(os.path.join(audio_out, f"{f_id}.json"), 'w', encoding='utf-8') as f:
            json.dump({"alignments": aligns}, f, ensure_ascii=False)

        manifest.append({"path": f"audio/{f_id}.wav", "duration": round(len(st_wav) / 24000, 2)})

    with open(os.path.join(output_root, "manifest.jsonl"), 'w', encoding='utf-8') as f:
        for e in manifest: f.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    INPUT_FOLDER = r"E:\Python_Project\dialouge test\StyleTTS2\claim_converation"
    OUTPUT_FOLDER = r"E:\Python_Project\dialouge test\Moshi_Dataset_Final2"
    process_moshi_dataset(INPUT_FOLDER, OUTPUT_FOLDER)