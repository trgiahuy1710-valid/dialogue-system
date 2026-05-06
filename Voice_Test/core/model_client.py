"""
model_client.py  —  fixed
─────────────────────────
Bug fixes:
  1. MockModelClient.transcribe_audio() bây giờ ĐỌC file thật (duration, path)
     và log rõ file nào đang được đọc.
  2. MockModelClient.generate_response() dùng nội dung transcript thật để
     quyết định scenario thay vì rotate mù _call_count.
  3. text_to_speech() guard: nếu text là JSON tool_call thì KHÔNG đọc, thay bằng
     câu thông báo, tránh TTS nhận raw JSON.
  4. Thêm [DEBUG] log chi tiết từng bước trong cả Mock lẫn QwenOmni.
"""

import json
import time
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("model_client")

# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────

import json as _json


def verify_policy(policy_id, customer_name):
    """Xác thực hợp đồng bảo hiểm."""
    if policy_id.startswith("HD"):
        return _json.dumps({
            "verification_status": "success",
            "policy_details": {
                "policy_id": policy_id,
                "customer_name": customer_name,
                "status": "active",
                "coverage_type": "medical_comprehensive",
            },
        }, ensure_ascii=False)
    return _json.dumps(
        {"verification_status": "failed", "reason": "Policy ID not found."},
        ensure_ascii=False,
    )


def process_medical_claim(policy_id, hospital_name, admission_date, diagnosis):
    """Khởi tạo hồ sơ bồi thường y tế."""
    return _json.dumps({
        "claim_status": "Pending Review",
        "required_documents": ["Giấy ra viện", "Hóa đơn VAT chi tiết", "Phiếu xét nghiệm"],
        "estimated_review_time": "5-7 business days",
        "ticket_id": f"MED-{policy_id[-4:]}",
    }, ensure_ascii=False)


def process_accident_claim(policy_id, incident_date, incident_description, injury_type):
    """Khởi tạo hồ sơ bồi thường tai nạn."""
    return _json.dumps({
        "claim_status": "Under Investigation",
        "incident_id": f"ACC-{policy_id[-4:]}",
        "action_required": "Cung cấp biên bản hiện trường từ công an",
    }, ensure_ascii=False)


def escalate_claim_discrepancy(claim_id, issue_type, request_manager):
    """Chuyển hồ sơ lên cấp quản lý."""
    status = "Thành công" if request_manager else "Chờ duyệt"
    return f"Hệ thống: Đã chuyển hồ sơ {claim_id} lý do '{issue_type}' lên cấp quản lý. Trạng thái: {status}."


AVAILABLE_FUNCTIONS = {
    "verify_policy": verify_policy,
    "process_medical_claim": process_medical_claim,
    "process_accident_claim": process_accident_claim,
    "escalate_claim_discrepancy": escalate_claim_discrepancy,
}

FUNCTIONS_LIST = [
    {
        "name": "verify_policy",
        "description": "Authenticate the insurance policy using the policy ID and the policyholder's full name.",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string", "description": "The insurance policy number (e.g., HD9988)"},
                "customer_name": {"type": "string", "description": "The full name of the customer"},
            },
            "required": ["policy_id", "customer_name"],
        },
    },
    {
        "name": "process_medical_claim",
        "description": "Initialize a claim ticket for medical conditions.",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "hospital_name": {"type": "string"},
                "admission_date": {"type": "string", "description": "ISO format: YYYY-MM-DD"},
                "diagnosis": {"type": "string"},
            },
            "required": ["policy_id", "hospital_name", "admission_date", "diagnosis"],
        },
    },
    {
        "name": "process_accident_claim",
        "description": "Initialize a claim ticket for an accident.",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "incident_date": {"type": "string", "description": "ISO format: YYYY-MM-DD"},
                "incident_description": {"type": "string"},
                "injury_type": {"type": "string"},
            },
            "required": ["policy_id", "incident_date", "incident_description", "injury_type"],
        },
    },
    {
        "name": "escalate_claim_discrepancy",
        "description": "Escalate insurance claim to manager.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "issue_type": {"type": "string", "enum": ["payout_discrepancy", "repair_vs_replacement"]},
                "request_manager": {"type": "boolean"},
            },
            "required": ["claim_id", "issue_type", "request_manager"],
        },
    },
]

SYSTEM_PROMPT = """Bạn là trợ lý AI chăm sóc khách hàng bảo hiểm. 
Bạn có khả năng suy nghĩ (thinking) trước khi trả lời, hãy đặt phần suy nghĩ trong <think>...</think>.
Bạn có quyền truy cập các công cụ sau để xử lý yêu cầu của khách hàng:

{tools_json}

Khi cần gọi tool, hãy trả về JSON dạng:
{{
  "role": "assistant",
  "content": [
    {{
      "type": "tool_call",
      "name": "<tên_tool>",
      "arguments": {{ ... }}
    }}
  ]
}}

Luôn xác minh hợp đồng trước khi xử lý bồi thường.
""".format(tools_json=json.dumps(FUNCTIONS_LIST, ensure_ascii=False, indent=2))


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class TurnMetrics:
    turn_id: str
    audio_file: str
    audio_duration_s: float = 0.0
    audio_encode_latency_s: float = 0.0
    inference_latency_s: float = 0.0
    tool_calls: list = field(default_factory=list)
    tool_latency_s: float = 0.0
    tts_latency_s: float = 0.0
    total_latency_s: float = 0.0
    thinker_text: str = ""
    response_text: str = ""
    response_audio_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        return {
            "turn_id": self.turn_id,
            "audio_file": self.audio_file,
            "audio_duration_s": round(self.audio_duration_s, 3),
            "audio_encode_latency_s": round(self.audio_encode_latency_s, 3),
            "inference_latency_s": round(self.inference_latency_s, 3),
            "tool_calls": self.tool_calls,
            "tool_latency_s": round(self.tool_latency_s, 3),
            "tts_latency_s": round(self.tts_latency_s, 3),
            "total_latency_s": round(self.total_latency_s, 3),
            "thinker_text": self.thinker_text,
            "response_text": self.response_text,
            "response_audio_path": self.response_audio_path,
            "error": self.error,
        }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_thinker(raw_text: str):
    """Tách <think>...</think> khỏi response. Trả về (thinker_text, clean_text)."""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(raw_text)
    thinker = "\n".join(m.strip() for m in matches)
    clean = pattern.sub("", raw_text).strip()
    return thinker, clean


def is_tool_call_json(text: str) -> bool:
    """Kiểm tra xem text có phải raw tool_call JSON không (để guard TTS)."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        obj = json.loads(stripped)
        content = obj.get("content", [])
        if isinstance(content, list):
            return any(c.get("type") == "tool_call" for c in content)
    except Exception:
        pass
    return False


def parse_tool_call(text: str):
    """Parse tool call từ response text. Trả về list[dict] hoặc []."""
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return []
    try:
        obj = json.loads(json_match.group())
        content = obj.get("content", [])
        if isinstance(content, list):
            return [c for c in content if c.get("type") == "tool_call"]
    except Exception:
        pass
    return []


def execute_tools(tool_calls: list) -> list:
    """Thực thi danh sách tool calls, trả về list results."""
    results = []
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("arguments", {})
        logger.debug(f"[TOOL] Execute: {name}({args})")
        if name in AVAILABLE_FUNCTIONS:
            try:
                result = AVAILABLE_FUNCTIONS[name](**args)
                results.append({"tool": name, "args": args, "result": result})
                logger.info(f"[TOOL] {name} → {result[:100] if isinstance(result, str) else result}")
            except Exception as e:
                results.append({"tool": name, "args": args, "error": str(e)})
                logger.error(f"[TOOL] {name} ERROR: {e}")
        else:
            results.append({"tool": name, "args": args, "error": f"Tool '{name}' not found"})
            logger.error(f"[TOOL] Unknown tool: {name}")
    return results


# ─────────────────────────────────────────────
# MOCK MODEL CLIENT
# ─────────────────────────────────────────────

# Keyword → scenario mapping để mock trả response hợp lý theo nội dung audio
_KEYWORD_MAP = [
    # (keywords_in_transcript, scenario_key)
    (["hợp đồng", "HD", "tên tôi là", "xác minh", "xác thực"], "verify"),
    (["nhập viện", "bệnh viện", "viêm", "điều trị", "phẫu thuật", "nằm viện"], "medical"),
    (["tai nạn", "ngã", "xe", "gãy", "thương tích", "va chạm"], "accident"),
    (["bất đồng", "không đồng ý", "sai", "khiếu nại", "leo thang", "cấp quản lý"], "escalate"),
]


def _detect_scenario(transcript: str) -> str:
    """Phân loại scenario từ transcript thật."""
    t_lower = transcript.lower()
    for keywords, scenario in _KEYWORD_MAP:
        if any(kw.lower() in t_lower for kw in keywords):
            logger.debug(f"[MockASR] Detected scenario='{scenario}' from transcript='{transcript[:60]}'")
            return scenario
    logger.debug(f"[MockASR] No keyword match → fallback='status' for: '{transcript[:60]}'")
    return "status"


class MockModelClient:
    """
    Mock client — KHÔNG cần GPU.
    Đọc metadata file thật, phân loại transcript, trả response có <think>.
    """

    def __init__(self, latency_range=(0.5, 1.5)):
        self.latency_range = latency_range
        logger.info("[MockModelClient] Initialized — reading REAL audio files (no GPU)")

    def transcribe_audio(self, audio_path: str) -> tuple[str, float]:
        """
        Giả lập ASR.
        - ĐỌC file thật để lấy duration
        - Dùng utterances.json/transcription.json nếu có để trả transcript thật
        - Fallback: sinh transcript giả dựa trên tên file
        """
        import random
        import soundfile as sf

        t0 = time.time()

        # ── Debug: xác nhận file tồn tại ──
        if not os.path.exists(audio_path):
            logger.error(f"[MockASR] FILE NOT FOUND: {audio_path}")
            return "[ERROR: file not found]", time.time() - t0

        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        logger.debug(f"[MockASR] READ: {audio_path} | sr={sr} | samples={len(audio)} | duration={duration:.2f}s")

        # ── Thử đọc transcript thật từ JSON trong cùng folder ──
        folder = os.path.dirname(audio_path)
        fname = os.path.basename(audio_path)  # e.g. user_0.wav
        turn_num = int(re.findall(r"\d+", fname)[0]) if re.findall(r"\d+", fname) else -1

        transcript = self._load_transcript_from_json(folder, fname, turn_num)

        if transcript:
            logger.info(f"[MockASR] Loaded from JSON: '{transcript}'")
        else:
            # Fallback: sinh transcript giả tương ứng với turn number
            fallback_pool = [
                "Xin chào, tôi muốn yêu cầu bồi thường bảo hiểm.",
                "Hợp đồng HD113734, tên tôi là Nguyễn Thị Mai.",
                "Tôi nhập viện tại Bệnh viện Bạch Mai ngày 2024-01-15 vì viêm phổi.",
                "Tôi bị tai nạn xe máy ngày 2024-02-10, gãy tay phải.",
                "Hồ sơ đã được xử lý chưa? Tôi cần theo dõi.",
            ]
            transcript = fallback_pool[turn_num % len(fallback_pool)]
            logger.warning(f"[MockASR] No JSON found → fallback transcript: '{transcript}'")

        time.sleep(random.uniform(0.1, 0.25))  # giả lập encode latency
        latency = time.time() - t0
        return transcript, latency

    def _load_transcript_from_json(self, folder: str, fname: str, turn_num: int) -> Optional[str]:
        """
        Đọc transcript thật từ transcription.json hoặc utterances.json.
        Tìm entry match với fname hoặc turn index.
        """
        for json_file in ["transcription.json", "utterances.json"]:
            jpath = os.path.join(folder, json_file)
            if not os.path.exists(jpath):
                continue
            try:
                with open(jpath, encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"[MockASR] Loaded {json_file} ({len(data)} entries)")

                # Tìm theo file name
                for entry in data:
                    if entry.get("file") == fname or entry.get("audio_file") == fname:
                        text = entry.get("text") or entry.get("transcript") or entry.get("content")
                        if text:
                            return str(text)

                # Tìm theo turn index
                for entry in data:
                    if entry.get("turn") == turn_num:
                        text = entry.get("text") or entry.get("transcript") or entry.get("content")
                        if text and entry.get("role", "user") == "user":
                            return str(text)

                # Tìm theo index trong list
                if isinstance(data, list) and 0 <= turn_num < len(data):
                    entry = data[turn_num]
                    text = entry.get("text") or entry.get("transcript") or entry.get("content")
                    if text:
                        logger.debug(f"[MockASR] Matched by list index [{turn_num}]")
                        return str(text)

            except Exception as e:
                logger.warning(f"[MockASR] Failed to parse {jpath}: {e}")

        return None

    def generate_response(self, messages: list, audio_path: str = None) -> tuple[str, float]:
        """
        Giả lập LLM inference.
        Dùng nội dung transcript cuối trong messages để chọn scenario.
        """
        import random

        t0 = time.time()
        delay = random.uniform(*self.latency_range)
        time.sleep(delay)

        # ── Lấy transcript từ message cuối cùng của user ──
        last_user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                last_user_text = content if isinstance(content, str) else str(content)
                break

        logger.debug(f"[MockLLM] Last user text: '{last_user_text[:100]}'")
        logger.debug(f"[MockLLM] History depth: {len(messages)} messages")
        logger.debug(f"[MockLLM] audio_path: {audio_path}")

        # Phân loại từ transcript thật
        scenario = _detect_scenario(last_user_text)

        # ── Sinh response theo scenario ──
        if scenario == "verify":
            # Trích policy_id và customer_name từ transcript
            policy_match = re.search(r"HD\d+", last_user_text, re.IGNORECASE)
            policy_id = policy_match.group() if policy_match else "HD000000"
            name_match = re.search(
                r"tên[^\w]+([\w\s]+?)(?:\.|,|$)", last_user_text, re.IGNORECASE
            )
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng"
            raw = f"""<think>
Người dùng cung cấp hợp đồng {policy_id} và tên {customer_name}.
Tôi cần gọi verify_policy để xác thực trước.
</think>
{{
  "role": "assistant",
  "content": [
    {{
      "type": "tool_call",
      "name": "verify_policy",
      "arguments": {{
        "policy_id": "{policy_id}",
        "customer_name": "{customer_name}"
      }}
    }}
  ]
}}"""

        elif scenario == "medical":
            # Trích thông tin nhập viện
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", last_user_text)
            admit_date = date_match.group() if date_match else "2024-01-01"
            hosp_match = re.search(r"bệnh viện[^\w]+([\w\s]+?)(?:\s+ngày|\s+vì|$)", last_user_text, re.IGNORECASE)
            hosp_name = hosp_match.group(1).strip() if hosp_match else "Bệnh viện"
            diag_match = re.search(r"vì\s+([\w\s]+?)(?:\.|,|$)", last_user_text, re.IGNORECASE)
            diagnosis = diag_match.group(1).strip() if diag_match else "Bệnh lý"
            raw = f"""<think>
Người dùng đề cập nhập viện tại "{hosp_name}" ngày {admit_date} vì "{diagnosis}".
Tôi cần tạo hồ sơ bồi thường y tế.
</think>
{{
  "role": "assistant",
  "content": [
    {{
      "type": "tool_call",
      "name": "process_medical_claim",
      "arguments": {{
        "policy_id": "HD000000",
        "hospital_name": "{hosp_name}",
        "admission_date": "{admit_date}",
        "diagnosis": "{diagnosis}"
      }}
    }}
  ]
}}"""

        elif scenario == "accident":
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", last_user_text)
            inc_date = date_match.group() if date_match else "2024-01-01"
            inj_match = re.search(r"gãy\s+([\w\s]+?)(?:\.|,|$)", last_user_text, re.IGNORECASE)
            injury = inj_match.group(1).strip() if inj_match else "Chấn thương"
            raw = f"""<think>
Người dùng bị tai nạn ngày {inc_date}, chấn thương: "{injury}".
Tôi cần tạo hồ sơ bồi thường tai nạn.
</think>
{{
  "role": "assistant",
  "content": [
    {{
      "type": "tool_call",
      "name": "process_accident_claim",
      "arguments": {{
        "policy_id": "HD000000",
        "incident_date": "{inc_date}",
        "incident_description": "{last_user_text[:100]}",
        "injury_type": "{injury}"
      }}
    }}
  ]
}}"""

        elif scenario == "escalate":
            raw = """<think>
Người dùng không đồng ý với kết quả bồi thường, cần chuyển lên manager.
</think>
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_call",
      "name": "escalate_claim_discrepancy",
      "arguments": {
        "claim_id": "MED-0000",
        "issue_type": "payout_discrepancy",
        "request_manager": true
      }
    }
  ]
}"""

        else:  # status / general
            raw = """<think>
Đây là câu hỏi thông thường về trạng thái hồ sơ.
Tôi sẽ trả lời trực tiếp, không cần gọi tool.
</think>
Xin chào! Hồ sơ bồi thường của bạn đang được xem xét. Thời gian xử lý dự kiến 5-7 ngày làm việc. Bạn có thể theo dõi qua app hoặc liên hệ hotline 1900-xxxx."""

        latency = time.time() - t0
        logger.debug(f"[MockLLM] Scenario='{scenario}' | latency={latency:.3f}s")
        return raw, latency

    def text_to_speech(self, text: str, output_path: str) -> tuple[str, float]:
        """
        Giả lập TTS.
        BUG FIX: guard against raw tool_call JSON being passed to TTS.
        """
        import numpy as np
        import soundfile as sf

        t0 = time.time()

        # ── Guard: nếu text là tool_call JSON → thay bằng thông báo ──
        if is_tool_call_json(text.strip()):
            logger.error(
                f"[MockTTS] BUG DETECTED: text is raw tool_call JSON — replacing with fallback!\n"
                f"  text preview: {text[:120]}"
            )
            text = "Yêu cầu của bạn đang được xử lý. Vui lòng chờ trong giây lát."

        if not text or not text.strip():
            logger.warning("[MockTTS] Empty text received — using fallback")
            text = "Xin lỗi, tôi không có phản hồi."

        logger.debug(f"[MockTTS] Synthesizing: '{text[:80]}' → {output_path}")
        time.sleep(0.08)

        # Tạo audio giả (sine wave ~1s)
        import numpy as np
        sr = 22050
        duration = min(max(len(text) * 0.05, 0.5), 3.0)
        t_arr = np.linspace(0, duration, int(sr * duration))
        audio = (0.2 * np.sin(2 * 3.14159 * 220 * t_arr)).astype(np.float32)

        sf.write(output_path, audio, sr)
        latency = time.time() - t0
        logger.debug(f"[MockTTS] Written {duration:.1f}s audio | latency={latency:.3f}s")
        return output_path, latency





# ─────────────────────────────────────────────
# REAL MODEL CLIENT (Qwen2.5-Omni via transformers)
# ─────────────────────────────────────────────

class QwenOmniClient:
    """Real client cho Qwen2.5-7B-Omni từ HuggingFace."""

    def __init__(self, model_name="thaomike/qwen-finetune-research", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        logger.info(f"[QwenOmni] Loading: {self.model_name} on {self.device}")
        t0 = time.time()
        try:
            from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
            self.model = Qwen2_5OmniModel.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=self.device,
            )
            self.model.eval()
            logger.info(f"[QwenOmni] Loaded in {time.time()-t0:.1f}s")
        except ImportError as e:
            logger.error(f"[QwenOmni] Import error: {e}")
            raise
        except Exception as e:
            logger.error(f"[QwenOmni] Load failed: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> tuple[str, float]:
        """ASR via Qwen2.5-Omni processor."""
        t0 = time.time()
        logger.debug(f"[QwenASR] Processing: {audio_path}")
        if not os.path.exists(audio_path):
            logger.error(f"[QwenASR] FILE NOT FOUND: {audio_path}")
            return "[ERROR: file not found]", 0.0

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Transcribe the following audio accurately in Vietnamese."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            },
        ]
        import torch
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256)
        transcript = self.processor.decode(out[0], skip_special_tokens=True)
        latency = time.time() - t0
        logger.debug(f"[QwenASR] → '{transcript}' ({latency:.3f}s)")
        return transcript, latency

    def generate_response(self, messages: list, audio_path: str = None) -> tuple[str, float]:
        """LLM inference với system prompt + tool schema."""
        t0 = time.time()
        logger.debug(f"[QwenLLM] history={len(messages)} msgs | audio={audio_path}")

        full_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ] + messages

        if audio_path and os.path.exists(audio_path):
            last = full_messages[-1]
            if isinstance(last["content"], str):
                last["content"] = [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": last["content"]},
                ]

        import torch
        inputs = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
            )
        raw = self.processor.decode(out[0], skip_special_tokens=True)
        latency = time.time() - t0
        logger.debug(f"[QwenLLM] raw preview: {raw[:120]} | latency={latency:.3f}s")
        return raw, latency

    def text_to_speech(self, text: str, output_path: str) -> tuple[str, float]:
        """TTS via Qwen2.5-Omni. Guard against tool_call JSON input."""
        t0 = time.time()

        # Guard
        if is_tool_call_json(text.strip()):
            logger.error(f"[QwenTTS] BUG: received raw tool_call JSON — replacing!")
            text = "Yêu cầu của bạn đang được xử lý. Vui lòng chờ trong giây lát."

        try:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Đọc to câu sau bằng tiếng Việt: {text}"}],
                }
            ]
            import torch
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_audio=True,
            ).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=2048)
            if hasattr(out, "audio"):
                import soundfile as sf
                sf.write(output_path, out.audio.cpu().numpy(), 24000)
            else:
                self._write_silence(output_path)
        except Exception as e:
            logger.warning(f"[QwenTTS] Error: {e} → writing silence")
            self._write_silence(output_path)

        latency = time.time() - t0
        return output_path, latency

    def _write_silence(self, path: str, duration: float = 1.0, sr: int = 16000):
        import numpy as np
        import soundfile as sf
        sf.write(path, np.zeros(int(sr * duration), dtype=np.float32), sr)