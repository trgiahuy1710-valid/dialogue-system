# VoiceTest System — Qwen2.5-7B-Omni Insurance Claim Tester

## Cấu trúc dự án

```
voicetest/
├── app.py                   # Gradio UI entry point
├── create_sample_data.py    # Tạo dataset mẫu để test
├── requirements.txt
├── core/
│   ├── model_client.py      # Model client (Mock + QwenOmni real)
│   └── pipeline.py          # Orchestrator + event system
└── logs/                    # Auto-created: log files + metrics JSON
    └── response_audio/      # TTS output files
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

Để dùng model thật (cần GPU):
```bash
pip install torch transformers accelerate
```

---

## Chạy nhanh với Mock Model (không cần GPU)

```bash
# Bước 1: Tạo sample data
python create_sample_data.py

# Bước 2: Chạy Gradio UI
python app.py --port 7860
```

Sau đó:
1. Mở `http://localhost:7860`
2. Root Directory: `./sample_data`
3. Tick **"Use Mock Model"**
4. Click **INITIALIZE PIPELINE**
5. Click **RUN**

---

## Chạy với dataset thật

```bash
python app.py --port 7860
```

Trong UI:
1. Root Directory: `/storage/huytg/dataset_audio_vieneu_clean_f5tts_v2`
2. Bỏ tick **"Use Mock Model"**
3. Model Name: `thaomike/qwen-finetune-research`
4. Initialize → Run

---

## Chạy CLI (không Gradio)

```bash
# Chạy tất cả conversations
python -c "
from core.pipeline import AudioTestPipeline

pipeline = AudioTestPipeline(
    root_dir='/storage/huytg/dataset_audio_vieneu_clean_f5tts_v2',
    log_dir='./logs',
    use_mock=True,
)
results = pipeline.run_all(start=0, end=10, max_files=3)
print(f'Done: {len(results)} turns')
"
```

---

## Debug từng phần

### Debug ASR:
```python
from core.model_client import MockModelClient
client = MockModelClient()
transcript, latency = client.transcribe_audio("path/to/user_0.wav")
print(f"Transcript: {transcript}")
print(f"Latency: {latency:.3f}s")
```

### Debug Thinker extraction:
```python
from core.model_client import extract_thinker
raw = "<think>Tôi cần xác minh hợp đồng trước.</think>Xin chào, tôi sẽ giúp bạn."
thinker, clean = extract_thinker(raw)
print("Thinker:", thinker)
print("Clean:", clean)
```

### Debug Tool call parser:
```python
from core.model_client import parse_tool_call, execute_tools
raw_json = '''{"role":"assistant","content":[{"type":"tool_call","name":"verify_policy","arguments":{"policy_id":"HD113734","customer_name":"Nguyễn Thị Mai"}}]}'''
calls = parse_tool_call(raw_json)
results = execute_tools(calls)
print(results)
```

---

## Metrics được log

| Metric | Mô tả |
|---|---|
| `audio_duration_s` | Độ dài file âm thanh đầu vào |
| `audio_encode_latency_s` | Thời gian ASR / audio encoding |
| `inference_latency_s` | Thời gian LLM inference (tổng, kể cả post-tool) |
| `tool_latency_s` | Thời gian thực thi tools |
| `tts_latency_s` | Thời gian text-to-speech |
| `total_latency_s` | End-to-end latency |
| `thinker_text` | Nội dung reasoning từ `<think>...</think>` |
| `tool_calls` | Danh sách tool được gọi |

---

## Tool System

Bốn tools được định nghĩa trong `core/model_client.py`:

| Tool | Chức năng |
|---|---|
| `verify_policy` | Xác thực hợp đồng bảo hiểm |
| `process_medical_claim` | Tạo hồ sơ bồi thường y tế |
| `process_accident_claim` | Tạo hồ sơ bồi thường tai nạn |
| `escalate_claim_discrepancy` | Chuyển hồ sơ lên cấp quản lý |

Model được prompt để trả JSON tool call format:
```json
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_call",
      "name": "verify_policy",
      "arguments": {
        "policy_id": "HD113734",
        "customer_name": "Nguyễn Thị Mai"
      }
    }
  ]
}
```

---

## Logs

Mỗi run tạo:
- `logs/run_YYYYMMDD_HHMMSS.log` — Full debug log
- `logs/metrics_gen-XXXX_YYYYMMDD.json` — Metrics JSON per conversation
- `logs/response_audio/*.wav` — TTS output files