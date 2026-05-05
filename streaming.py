import os
import re
import soundfile as sf
import time
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Streaming Simulator")
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder chứa gen-xxxx")
    parser.add_argument("--start", type=int, default=0,
                        help="Bắt đầu từ gen-xxxx")
    parser.add_argument("--end", type=int, default=None,
                        help="Kết thúc gen-xxxx")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Giới hạn số file user mỗi conversation")
    parser.add_argument("--chunk_ms", type=int, default=20,
                        help="Chunk size (ms)")
    parser.add_argument("--debug", action="store_true",
                        help="Bật debug log")
    parser.add_argument("--realtime", action="store_true",
                        help="Giả lập realtime (sleep)")
    return parser.parse_args()

def get_gen_folders(root):
    folders = os.listdir(root)
    gen_folders = [f for f in folders if f.startswith("gen-")]
    # sort theo số
    gen_folders = sorted(gen_folders, key=lambda x: int(re.findall(r'\d+', x)[0]))

    return gen_folders

def filter_gen_folders(gen_folders, start, end):
    result = []
    for f in gen_folders:
        num = int(re.findall(r'\d+', f)[0])

        if num >= start and (end is None or num <= end):
            result.append(f)

    return result

def get_user_files(folder):
    files = os.listdir(folder)
    user_files = [f for f in files if f.startswith("user_") and f.endswith(".wav")]
    # sort theo số
    user_files = sorted(user_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    return user_files

CHUNK_MS = 20  # giả lập realtime

def stream_audio(file_path):
    audio, sr = sf.read(file_path)

    chunk_size = int(sr * CHUNK_MS / 1000)
    total_chunks = len(audio) // chunk_size

    print(f"\n=== Streaming: {os.path.basename(file_path)} ===")
    print(f"Sample rate: {sr}, Total chunks: {total_chunks}")

    start_time = time.time()
    prev_time = start_time

    total = len(audio) // chunk_size

    for i in tqdm(range(0, len(audio), chunk_size), desc="Streaming", unit="chunk"):
        chunk = audio[i:i+chunk_size]

        now = time.time()
        delta = now - prev_time

        # print(f"[DEBUG] chunk={i//chunk_size:03d} | size={len(chunk)} | Δt={delta:.4f}s")
        prev_time = now

        # giả lập realtime
        time.sleep(CHUNK_MS / 1000)

    end_time = time.time()

    audio_duration = len(audio) / sr
    real_time = end_time - start_time

    print(f"[SUMMARY]")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Streaming time: {real_time:.2f}s")
    print(f"Realtime ratio: {real_time/audio_duration:.2f}")

def progress_bar(i, total, length=30):
    percent = i / total
    bar = "█" * int(percent * length)
    return f"[{bar:<30}] {percent*100:.1f}%"

def run_conversation(folder):
    files = os.listdir(folder)

    user_files = [f for f in files if f.startswith("user_") and f.endswith(".wav")]
    user_files = sorted(user_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    print("User files:", user_files)

    for f in user_files:
        stream_audio(os.path.join(folder, f))

if __name__ == "__main__":
    args = parse_args()
    run_conversation(args.root)