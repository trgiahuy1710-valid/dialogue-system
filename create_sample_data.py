"""
create_sample_data.py
─────────────────────
Tạo sample gen-xxxx folders với user_x.wav giả để test pipeline 
mà không cần dataset thật.
"""

import os
import json
import numpy as np
import soundfile as sf

def create_sample_conversation(folder_path: str, n_turns: int = 4, sr: int = 16000):
    os.makedirs(folder_path, exist_ok=True)

    # Mô phỏng conversation insurance claim
    scenario = [
        (0, "user", "Xin chào, tôi muốn yêu cầu bồi thường bảo hiểm y tế."),
        (2, "user", "Hợp đồng của tôi là HD113734, tên tôi là Nguyễn Thị Mai."),
        (4, "user", "Tôi nhập viện tại Bệnh viện Bạch Mai ngày 2024-01-15 vì viêm phổi."),
        (6, "user", "Hồ sơ đã được tạo chưa? Tôi cần theo dõi."),
    ]

    transcriptions = []
    for turn_idx, (audio_idx, role, text) in enumerate(scenario[:n_turns]):
        # Tạo audio sine wave ngắn để giả lập giọng nói
        duration = 2.0 + turn_idx * 0.5  # 2s, 2.5s, 3s, 3.5s
        t = np.linspace(0, duration, int(sr * duration))
        freq = 200 + turn_idx * 50  # khác nhau để phân biệt
        audio = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        # Thêm noise nhỏ
        audio += 0.02 * np.random.randn(len(audio)).astype(np.float32)

        fname = f"user_{audio_idx}.wav"
        sf.write(os.path.join(folder_path, fname), audio, sr)
        transcriptions.append({
            "file": fname,
            "role": role,
            "text": text,
            "duration_s": duration
        })

    # Tạo transcription.json và utterances.json
    with open(os.path.join(folder_path, "transcription.json"), "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    utterances = [{"turn": i, "text": t[2], "role": t[1]} for i, t in enumerate(scenario[:n_turns])]
    with open(os.path.join(folder_path, "utterances.json"), "w", encoding="utf-8") as f:
        json.dump(utterances, f, ensure_ascii=False, indent=2)

    print(f"  Created: {folder_path} ({n_turns} turns)")

def create_sample_dataset(root: str = "./sample_data", n_conversations: int = 5):
    print(f"Creating sample dataset at: {root}")
    os.makedirs(root, exist_ok=True)

    for i in range(n_conversations):
        folder_name = f"gen-{i:04d}"
        folder_path = os.path.join(root, folder_name)
        n_turns = 3 + (i % 3)  # 3-5 turns
        create_sample_conversation(folder_path, n_turns)

    print(f"\n✔ Created {n_conversations} conversations in {root}")
    print("Contents:")
    for item in sorted(os.listdir(root)):
        sub = os.path.join(root, item)
        files = os.listdir(sub)
        print(f"  {item}/  ({len(files)} files)")

if __name__ == "__main__":
    create_sample_dataset()