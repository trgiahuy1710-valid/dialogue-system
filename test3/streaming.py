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

def stream_audio(file_path, chunk_ms=20, debug=False, realtime=True):
    audio, sr = sf.read(file_path)

    chunk_size = int(sr * chunk_ms / 1000)

    print(f"\n=== Streaming: {os.path.basename(file_path)} ===")

    start_time = time.time()
    prev_time = start_time

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]

        now = time.time()
        delta = now - prev_time

        if debug:
            print(f"[DEBUG] chunk={i//chunk_size:03d} | Δt={delta:.4f}s")

        prev_time = now

        if realtime:
            time.sleep(chunk_ms / 1000)

    end_time = time.time()

    audio_duration = len(audio) / sr
    real_time = end_time - start_time

    print(f"[SUMMARY] duration={audio_duration:.2f}s | stream={real_time:.2f}s | ratio={real_time/audio_duration:.2f}")

def progress_bar(i, total, length=30):
    percent = i / total
    bar = "█" * int(percent * length)
    return f"[{bar:<30}] {percent*100:.1f}%"

def run_conversation(folder, max_files=None, **kwargs):
    user_files = get_user_files(folder)

    if max_files:
        user_files = user_files[:max_files]

    print(f"\n📁 {folder}")
    print("User files:", user_files)

    for f in user_files:
        stream_audio(os.path.join(folder, f), **kwargs)

if __name__ == "__main__":
    args = parse_args()

    gen_folders = get_gen_folders(args.root)
    gen_folders = filter_gen_folders(gen_folders, args.start, args.end)

    print("Selected folders:", gen_folders)

    for folder in gen_folders:
        full_path = os.path.join(args.root, folder)

        run_conversation(
            full_path,
            max_files=args.max_files,
            chunk_ms=args.chunk_ms,
            debug=args.debug,
            realtime=args.realtime
        )