import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from munch import Munch
from nltk.tokenize import sent_tokenize, word_tokenize

os.environ.setdefault("PHONEMIZER_ESPEAK_EXECUTABLE", r"C:\Program Files\eSpeak NG\espeak-ng.exe")
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", r"C:\Program Files\eSpeak NG\libespeak-ng.dll")

import phonemizer

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule


LOGGER = logging.getLogger("create_omni_dataset_styletts2")
SYSTEM_PROMPT = (
    "Bạn là trợ lý bảo hiểm. Hãy dựa trên hội thoại và audio của khách hàng để trả lời. "
    "Nếu đủ thông tin thì gọi tool. Nếu chưa đủ thì hỏi tiếp."
)

INVISIBLE_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
MULTISPACE_PATTERN = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([,.;:!?%/\]\)])")
SPACE_AFTER_OPEN_PATTERN = re.compile(r"([\[\(])\s+")
REPEATED_PUNCT_PATTERN = re.compile(r"([,;:!?])\1+")
SENTENCE_FALLBACK_SPLIT = re.compile(r"(?<=[.!?;:])\s+")

SAMPLE_RATE = 24000
MEL_MEAN = -4
MEL_STD = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
)
textcleaner = TextCleaner()
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80,
    n_fft=2048,
    win_length=1200,
    hop_length=300,
)

MODEL_CACHE: Dict[str, Any] = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tao omni dataset bang StyleTTS2, giu 2 giong noi co dinh tu 2 reference audio."
    )
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--input", type=str, default=str(base_dir / "standard_flow_en_fn.jsonl"))
    parser.add_argument("--output", type=str, default=str(base_dir / "omni_train_styletts2.jsonl"))
    parser.add_argument("--audio-dir", type=str, default=str(base_dir / "dataset_audio_styletts2"))
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--user-reference",
        type=str,
        default=str(base_dir / "Demo" / "reference_audio" / "Gavin.wav"),
    )
    parser.add_argument(
        "--assistant-reference",
        type=str,
        default=str(base_dir / "Demo" / "reference_audio" / "1221-135767-0014.wav"),
    )
    parser.add_argument("--config-path", type=str, default=str(base_dir / "Models" / "LibriTTS" / "config.yml"))
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=str(base_dir / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"),
    )
    parser.add_argument("--segment-max-chars", type=int, default=220)
    parser.add_argument("--join-gap-ms", type=int, default=35)
    parser.add_argument("--leading-trim-ms", type=int, default=20)
    parser.add_argument("--trailing-trim-ms", type=int, default=60)
    parser.add_argument("--min-rms", type=float, default=0.0015)
    parser.add_argument("--warn-seconds", type=float, default=30.0)
    parser.add_argument("--user-alpha", type=float, default=0.2)
    parser.add_argument("--user-beta", type=float, default=0.8)
    parser.add_argument("--assistant-alpha", type=float, default=0.7)
    parser.add_argument("--assistant-beta", type=float, default=0.3)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--embedding-scale", type=float, default=1.0)
    return parser.parse_args()


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    LOGGER.info("Log file: %s", log_file)


def append_jsonl(log_file: str, payload: Dict[str, Any]) -> None:
    record = dict(payload)
    record.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_log_file(audio_dir: str, log_file: Optional[str]) -> str:
    if log_file:
        return log_file
    return str(Path(audio_dir) / "create_dataset_styletts2.log.jsonl")


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch({k: recursive_munch(v) for k, v in d.items()})
    if isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1)
    mask = mask.type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def preprocess(wave: np.ndarray) -> torch.Tensor:
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEL_MEAN) / MEL_STD
    return mel_tensor


def normalize_text(text: str) -> Tuple[str, Dict[str, Any]]:
    original = "" if text is None else str(text)
    cleaned = unicodedata.normalize("NFC", original)
    cleaned = INVISIBLE_PATTERN.sub(" ", cleaned)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\t": " ",
        "\r": " ",
        "\n": " ",
    }
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()
    cleaned = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", cleaned)
    cleaned = SPACE_AFTER_OPEN_PATTERN.sub(r"\1", cleaned)
    cleaned = REPEATED_PUNCT_PATTERN.sub(r"\1", cleaned)
    cleaned = cleaned.strip()
    meta = {
        "original_len": len(original),
        "normalized_len": len(cleaned),
        "changed": cleaned != original,
    }
    return cleaned, meta


def sentence_chunks(text: str, max_chars: int) -> List[str]:
    normalized, _ = normalize_text(text)
    if not normalized:
        return []

    try:
        sentences = sent_tokenize(normalized)
    except LookupError:
        sentences = SENTENCE_FALLBACK_SPLIT.split(normalized)

    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [normalized]

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sentence) <= max_chars:
            current = sentence
            continue
        words = sentence.split()
        small = ""
        for word in words:
            word_candidate = f"{small} {word}".strip() if small else word
            if len(word_candidate) <= max_chars:
                small = word_candidate
                continue
            if small:
                chunks.append(small)
                small = ""
            if len(word) <= max_chars:
                small = word
            else:
                for start in range(0, len(word), max_chars):
                    chunks.append(word[start : start + max_chars])
        if small:
            current = small
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).flatten()
    if arr.size == 0:
        return arr
    return arr - np.mean(arr, dtype=np.float32)


def frame_rms(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size < frame_length:
        return np.asarray([float(np.sqrt(np.mean(np.square(audio))))], dtype=np.float32)
    values: List[float] = []
    for start in range(0, audio.size - frame_length + 1, hop_length):
        frame = audio[start : start + frame_length]
        values.append(float(np.sqrt(np.mean(np.square(frame))) + 1e-12))
    return np.asarray(values, dtype=np.float32)


def find_active_bounds(audio: np.ndarray, sample_rate: int, min_rms: float) -> Tuple[int, int, Dict[str, Any]]:
    arr = np.asarray(audio, dtype=np.float32).flatten()
    info = {"start_sec": 0.0, "end_sec": round(arr.size / sample_rate, 4) if sample_rate > 0 else 0.0}
    if arr.size == 0 or sample_rate <= 0:
        return 0, arr.size, info

    frame_length = max(1, int(sample_rate * 0.02))
    hop_length = max(1, int(sample_rate * 0.01))
    rms = frame_rms(arr, frame_length, hop_length)
    if rms.size == 0:
        return 0, arr.size, info

    head = rms[: min(8, rms.size)]
    tail = rms[max(0, rms.size - 8) :]
    noise_floor = float(np.median(np.concatenate([head, tail]))) if rms.size else 0.0
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    threshold = max(min_rms, noise_floor * 3.0, peak * 0.018)

    active = rms >= threshold
    if not np.any(active):
        return 0, arr.size, info

    start_frame = int(np.argmax(active))
    end_frame = len(active) - 1 - int(np.argmax(active[::-1]))
    start_sample = max(0, start_frame * hop_length)
    end_sample = min(arr.size, end_frame * hop_length + frame_length)

    info["start_sec"] = round(start_sample / sample_rate, 4)
    info["end_sec"] = round(end_sample / sample_rate, 4)
    info["threshold"] = round(threshold, 6)
    info["noise_floor"] = round(noise_floor, 6)
    return start_sample, end_sample, info


def trim_audio_edges(
    audio: np.ndarray,
    sample_rate: int,
    leading_trim_ms: int,
    trailing_trim_ms: int,
    min_rms: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = remove_dc_offset(audio)
    if arr.size == 0 or sample_rate <= 0:
        return arr, {"trimmed_start_samples": 0, "trimmed_end_samples": 0}

    start_sample, end_sample, bounds = find_active_bounds(arr, sample_rate, min_rms)
    keep_start = max(0, start_sample - int(sample_rate * (leading_trim_ms / 1000.0)))
    keep_end = min(arr.size, end_sample + int(sample_rate * (trailing_trim_ms / 1000.0)))
    trimmed = arr[keep_start:keep_end].copy()
    if trimmed.size == 0:
        trimmed = arr.copy()

    fade_in = min(int(sample_rate * 0.008), trimmed.size)
    fade_out = min(int(sample_rate * 0.012), trimmed.size)
    if fade_in > 1:
        trimmed[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 1:
        trimmed[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)

    info = {
        "trimmed_start_samples": int(keep_start),
        "trimmed_end_samples": int(arr.size - keep_end),
        "trimmed_start_sec": round(keep_start / sample_rate, 4),
        "trimmed_end_sec": round((arr.size - keep_end) / sample_rate, 4),
        "speech_start_sec": bounds.get("start_sec", 0.0),
        "speech_end_sec": bounds.get("end_sec", 0.0),
        "rms_threshold": bounds.get("threshold", 0.0),
        "noise_floor": bounds.get("noise_floor", 0.0),
    }
    return trimmed, info


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).flatten()
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    if peak <= 0:
        return arr
    if peak > 0.97:
        arr = arr / peak * 0.97
    return arr


def analyze_audio(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    arr = np.asarray(audio, dtype=np.float32).flatten()
    duration = round(arr.size / sample_rate, 3) if sample_rate > 0 else 0.0
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    leading = arr[: min(arr.size, max(1, int(sample_rate * 0.25)))]
    leading_rms = float(np.sqrt(np.mean(np.square(leading)))) if leading.size else 0.0
    return {
        "num_samples": int(arr.size),
        "duration_seconds": duration,
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "leading_rms": round(leading_rms, 6),
    }


def save_audio(audio_path: str, audio: np.ndarray) -> None:
    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE)


def compute_style(path: str) -> torch.Tensor:
    wave, sr = librosa.load(path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    mel_tensor = preprocess(audio).to(device)
    model = MODEL_CACHE["model"]
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def load_styletts2(args) -> None:
    if MODEL_CACHE:
        return

    config = yaml.safe_load(open(args.config_path, "r", encoding="utf-8"))
    text_aligner = load_ASR_models(config.get("ASR_path"), config.get("ASR_config"))
    pitch_extractor = load_F0_models(config.get("F0_path"))
    plbert = load_plbert(config.get("PLBERT_dir"))

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    params = torch.load(args.checkpoint_path, map_location="cpu")["net"]
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except Exception:
                from collections import OrderedDict

                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k[7:]] = v
                model[key].load_state_dict(new_state_dict, strict=False)
        model[key].eval().to(device)

    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )
    MODEL_CACHE["model"] = model
    MODEL_CACHE["sampler"] = sampler


def inference(text: str, ref_s: torch.Tensor, alpha: float, beta: float, diffusion_steps: int, embedding_scale: float):
    text = text.strip()
    if not text:
        return np.zeros(0, dtype=np.float32)

    phonemes = global_phonemizer.phonemize([text])[0]
    tokens = textcleaner(" ".join(word_tokenize(phonemes)))
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    model = MODEL_CACHE["model"]
    sampler = MODEL_CACHE["sampler"]

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = beta * s_pred[:, 128:] + (1 - beta) * ref_s[:, 128:]
        ref = alpha * s_pred[:, :128] + (1 - alpha) * ref_s[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln.size(0)):
            pred_aln[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln.unsqueeze(0).to(device)
        f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder(t_en @ pred_aln.unsqueeze(0).to(device), f0_pred, n_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50]


def synthesize_styletts2(text: str, ref_s: torch.Tensor, args, alpha: float, beta: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    chunks = sentence_chunks(text, args.segment_max_chars)
    if not chunks:
        return np.array([], dtype=np.float32), {"segments": [], "segment_count": 0}

    segment_reports: List[Dict[str, Any]] = []
    rendered_segments: List[np.ndarray] = []
    gap = np.zeros(int(SAMPLE_RATE * (args.join_gap_ms / 1000.0)), dtype=np.float32)

    for idx, chunk in enumerate(chunks):
        raw_audio = inference(
            text=chunk,
            ref_s=ref_s,
            alpha=alpha,
            beta=beta,
            diffusion_steps=args.diffusion_steps,
            embedding_scale=args.embedding_scale,
        )
        raw_audio = np.asarray(raw_audio, dtype=np.float32).flatten()
        trimmed_audio, trim_info = trim_audio_edges(
            raw_audio,
            SAMPLE_RATE,
            args.leading_trim_ms,
            args.trailing_trim_ms,
            args.min_rms,
        )
        if trimmed_audio.size:
            rendered_segments.append(trimmed_audio)
        if idx < len(chunks) - 1 and gap.size:
            rendered_segments.append(gap.copy())
        segment_reports.append(
            {
                "segment_index": idx,
                "text": chunk,
                "text_len": len(chunk),
                "raw_samples": int(raw_audio.size),
                "trimmed_samples": int(trimmed_audio.size),
                "trim": trim_info,
            }
        )

    final_audio = np.concatenate(rendered_segments) if rendered_segments else np.array([], dtype=np.float32)
    final_audio = normalize_audio(final_audio)
    return final_audio, {"segments": segment_reports, "segment_count": len(segment_reports)}


def build_audio_message(role: str, content: str, attach_audio: bool) -> Dict[str, Any]:
    if attach_audio:
        return {"role": role, "content": f"{content} <audio>"}
    return {"role": role, "content": content}


def process_message_audio(
    content: str,
    ref_s: torch.Tensor,
    audio_path: str,
    args,
    log_file: str,
    context: Dict[str, Any],
    alpha: float,
    beta: float,
) -> Tuple[bool, Dict[str, Any]]:
    started_at = time.perf_counter()
    tts_text, text_meta = normalize_text(content)
    if not tts_text:
        result = {"error": "empty_tts_text", "text_meta": text_meta}
        append_jsonl(log_file, {**context, "event": "synth_skip", **result})
        return False, result

    try:
        audio, synth_meta = synthesize_styletts2(tts_text, ref_s, args, alpha, beta)
        metrics = analyze_audio(audio, SAMPLE_RATE)
        metrics["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        if metrics["num_samples"] == 0:
            raise ValueError("empty_audio_after_synthesis")
        if metrics["duration_seconds"] > args.warn_seconds:
            LOGGER.warning("[%s] utterance dai %.2fs: %s", context["conv_id"], metrics["duration_seconds"], audio_path)
        if metrics["leading_rms"] > max(metrics["rms"] * 2.5, 0.08):
            LOGGER.warning("[%s] leading RMS van con cao, can nghe kiem tra file %s", context["conv_id"], audio_path)
        save_audio(audio_path, audio)
        result = {
            "tts_text": tts_text,
            "text_meta": text_meta,
            "synth_meta": synth_meta,
            "metrics": metrics,
        }
        append_jsonl(log_file, {**context, "event": "synth_ok", **result})
        return True, result
    except Exception as exc:
        result = {
            "error": repr(exc),
            "tts_text": tts_text,
            "text_meta": text_meta,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }
        append_jsonl(log_file, {**context, "event": "synth_failed", **result})
        return False, result


def main() -> None:
    args = parse_args()
    os.makedirs(args.audio_dir, exist_ok=True)
    log_file = resolve_log_file(args.audio_dir, args.log_file)
    setup_logging(log_file)

    LOGGER.info("Input: %s", args.input)
    LOGGER.info("Output: %s", args.output)
    LOGGER.info("Audio dir: %s", args.audio_dir)
    LOGGER.info("Device: %s", device)

    load_styletts2(args)
    ref_user = compute_style(args.user_reference)
    ref_assistant = compute_style(args.assistant_reference)

    append_jsonl(
        log_file,
        {
            "event": "run_start",
            "input": args.input,
            "output": args.output,
            "audio_dir": args.audio_dir,
            "config_path": args.config_path,
            "checkpoint_path": args.checkpoint_path,
            "user_reference": args.user_reference,
            "assistant_reference": args.assistant_reference,
            "start": args.start,
            "end": args.end,
            "diffusion_steps": args.diffusion_steps,
            "embedding_scale": args.embedding_scale,
            "user_alpha": args.user_alpha,
            "user_beta": args.user_beta,
            "assistant_alpha": args.assistant_alpha,
            "assistant_beta": args.assistant_beta,
        },
    )

    write_mode = "a" if args.start > 1 else "w"
    stats = {
        "conversations_total": 0,
        "messages_total": 0,
        "audio_ok": 0,
        "audio_failed": 0,
    }

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, write_mode, encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            if line_num < args.start:
                continue
            if args.end and line_num > args.end:
                break

            record = json.loads(line)
            conv_id = record.get("id", f"gen-{line_num:04d}")
            messages = record.get("messages", [])
            stats["conversations_total"] += 1

            conv_audio_dir = os.path.join(args.audio_dir, conv_id)
            os.makedirs(conv_audio_dir, exist_ok=True)

            LOGGER.info("[#%s | %s] messages=%s", line_num, conv_id, len(messages))

            omni_messages: List[Dict[str, Any]] = []
            omni_audios: List[str] = []
            transcript_messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            transcript_audios: List[str] = []
            utterance_report: List[Dict[str, Any]] = []

            for msg_idx, msg in enumerate(messages):
                stats["messages_total"] += 1
                role = msg.get("role")
                content = msg.get("content", "")

                if role not in {"user", "assistant"}:
                    omni_messages.append(msg)
                    transcript_messages.append(msg)
                    continue

                if not isinstance(content, str) or not content.strip():
                    omni_messages.append({"role": role, "content": content})
                    transcript_messages.append({"role": role, "content": content})
                    continue

                is_user = role == "user"
                ref_s = ref_user if is_user else ref_assistant
                alpha = args.user_alpha if is_user else args.assistant_alpha
                beta = args.user_beta if is_user else args.assistant_beta

                audio_filename = f"{role}_{msg_idx}.wav"
                audio_path = os.path.join(conv_audio_dir, audio_filename)
                context = {
                    "line_num": line_num,
                    "conv_id": conv_id,
                    "msg_idx": msg_idx,
                    "role": role,
                    "audio_path": audio_path,
                    "reference_audio": args.user_reference if is_user else args.assistant_reference,
                }

                ok, result = process_message_audio(
                    content=content,
                    ref_s=ref_s,
                    audio_path=audio_path,
                    args=args,
                    log_file=log_file,
                    context=context,
                    alpha=alpha,
                    beta=beta,
                )

                if ok:
                    stats["audio_ok"] += 1
                    omni_messages.append(build_audio_message(role, content, attach_audio=True))
                    omni_audios.append(audio_path)

                    if is_user:
                        transcript_messages.append(build_audio_message(role, content, attach_audio=True))
                        transcript_audios.append(audio_filename)
                    else:
                        transcript_messages.append({"role": role, "content": content})
                else:
                    stats["audio_failed"] += 1
                    omni_messages.append({"role": role, "content": content})
                    transcript_messages.append({"role": role, "content": content})

                utterance_report.append(
                    {
                        "msg_idx": msg_idx,
                        "role": role,
                        "audio": audio_filename if ok else None,
                        "reference_audio": context["reference_audio"],
                        "source_text": content,
                        "tts_text": result.get("tts_text") if isinstance(result, dict) else None,
                        "text_meta": result.get("text_meta") if isinstance(result, dict) else None,
                        "metrics": result.get("metrics") if isinstance(result, dict) else None,
                        "error": result.get("error") if isinstance(result, dict) else None,
                    }
                )

            omni_record: Dict[str, Any] = {"messages": omni_messages}
            if omni_audios:
                omni_record["audios"] = omni_audios
            fout.write(json.dumps(omni_record, ensure_ascii=False) + "\n")
            fout.flush()

            transcript_record: Dict[str, Any] = {"messages": transcript_messages}
            if transcript_audios:
                transcript_record["audios"] = transcript_audios
            with open(os.path.join(conv_audio_dir, "transcription.json"), "w", encoding="utf-8") as tf:
                json.dump(transcript_record, tf, ensure_ascii=False, indent=2)

            with open(os.path.join(conv_audio_dir, "utterances.json"), "w", encoding="utf-8") as uf:
                json.dump(
                    {
                        "conv_id": conv_id,
                        "user_reference": args.user_reference,
                        "assistant_reference": args.assistant_reference,
                        "utterances": utterance_report,
                    },
                    uf,
                    ensure_ascii=False,
                    indent=2,
                )

    append_jsonl(log_file, {"event": "run_done", "stats": stats})
    LOGGER.info("Done. Stats: %s", stats)


if __name__ == "__main__":
    main()
