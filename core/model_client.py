"""
model_client.py
───────────────
Client giao tiếp với Qwen2.5-7B-Omni qua API hoặc local transformers.
Xử lý:
  - Tool calling (verify_policy, process_medical_claim, ...)
  - Thinker / reasoning extraction
  - Latency logging chi tiết
"""

import json
import time
import logging
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
                "coverage_type": "medical_comprehensive"
            }
        }, ensure_ascii=False)
    return _json.dumps({"verification_status": "failed", "reason": "Policy ID not found."}, ensure_ascii=False)

def process_medical_claim(policy_id, hospital_name, admission_date, diagnosis):
    """Khởi tạo hồ sơ bồi thường y tế."""
    return _json.dumps({
        "claim_status": "Pending Review",
        "required_documents": ["Giấy ra viện", "Hóa đơn VAT chi tiết", "Phiếu xét nghiệm"],
        "estimated_review_time": "5-7 business days",
        "ticket_id": f"MED-{policy_id[-4:]}"
    }, ensure_ascii=False)

def process_accident_claim(policy_id, incident_date, incident_description, injury_type):
    """Khởi tạo hồ sơ bồi thường tai nạn."""
    return _json.dumps({
        "claim_status": "Under Investigation",
        "incident_id": f"ACC-{policy_id[-4:]}",
        "action_required": "Cung cấp biên bản hiện trường từ công an"
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
                "customer_name": {"type": "string", "description": "The full name of the customer"}
            },
            "required": ["policy_id", "customer_name"]
        }
    },
    {
        "name": "process_medical_claim",
        "description": "Initialize a claim ticket for medical conditions (hospitalization, inpatient/outpatient treatment due to illness).",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "hospital_name": {"type": "string"},
                "admission_date": {"type": "string", "description": "ISO format: YYYY-MM-DD"},
                "diagnosis": {"type": "string", "description": "Description of the medical condition or diagnosis"}
            },
            "required": ["policy_id", "hospital_name", "admission_date", "diagnosis"]
        }
    },
    {
        "name": "process_accident_claim",
        "description": "Initialize a claim ticket for an accident (e.g., falling off a bike, collision, physical injury).",
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "incident_date": {"type": "string", "description": "ISO format: YYYY-MM-DD"},
                "incident_description": {"type": "string"},
                "injury_type": {"type": "string"}
            },
            "required": ["policy_id", "incident_date", "incident_description", "injury_type"]
        }
    },
    {
        "name": "escalate_claim_discrepancy",
        "description": "Escalate insurance claim to manager.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "issue_type": {"type": "string", "enum": ["payout_discrepancy", "repair_vs_replacement"]},
                "request_manager": {"type": "boolean"}
            },
            "required": ["claim_id", "issue_type", "request_manager"]
        }
    }
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
    # ASR / audio encode
    audio_encode_latency_s: float = 0.0
    # Model inference
    inference_latency_s: float = 0.0
    # Tool execution (nếu có)
    tool_calls: list = field(default_factory=list)
    tool_latency_s: float = 0.0
    # TTS
    tts_latency_s: float = 0.0
    # End-to-end
    total_latency_s: float = 0.0
    # Thinker
    thinker_text: str = ""
    # Response text
    response_text: str = ""
    # Audio response path
    response_audio_path: Optional[str] = None
    # Error
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
# THINKER EXTRACTOR
# ─────────────────────────────────────────────

def extract_thinker(raw_text: str):
    """
    Tách phần <think>...</think> khỏi response.
    Trả về (thinker_text, clean_text)
    """
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(raw_text)
    thinker = "\n".join(m.strip() for m in matches)
    clean = pattern.sub("", raw_text).strip()
    return thinker, clean


# ─────────────────────────────────────────────
# TOOL CALL PARSER
# ─────────────────────────────────────────────

def parse_tool_call(text: str):
    """
    Parse tool call từ response text.
    Hỗ trợ cả JSON block và inline JSON.
    Trả về list[dict] hoặc []
    """
    # Tìm JSON block đầu tiên
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
    """
    Thực thi danh sách tool calls, trả về list results.
    """
    results = []
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("arguments", {})
        if name in AVAILABLE_FUNCTIONS:
            try:
                result = AVAILABLE_FUNCTIONS[name](**args)
                results.append({"tool": name, "args": args, "result": result})
                logger.debug(f"[TOOL] {name}({args}) → {result}")
            except Exception as e:
                results.append({"tool": name, "args": args, "error": str(e)})
                logger.error(f"[TOOL ERROR] {name}: {e}")
        else:
            results.append({"tool": name, "args": args, "error": "Tool not found"})
    return results


# ─────────────────────────────────────────────
# MOCK MODEL CLIENT (dùng khi chưa có GPU / model)
# ─────────────────────────────────────────────

class MockModelClient:
    """
    Mock client để test pipeline mà không cần GPU.
    Sinh ra các response giả lập có think block và tool calls.
    """
    def __init__(self, latency_range=(0.5, 2.0)):
        self.latency_range = latency_range
        self._call_count = 0
        logger.info("[MockModelClient] Initialized (no GPU needed)")

    def transcribe_audio(self, audio_path: str) -> tuple[str, float]:
        """Giả lập ASR. Returns (transcript, latency_s)"""
        import random
        t0 = time.time()
        time.sleep(random.uniform(0.1, 0.3))
        samples = [
            "Xin chào, tôi muốn yêu cầu bồi thường bảo hiểm.",
            "Hợp đồng của tôi là HD113734, tên tôi là Nguyễn Thị Mai.",
            "Tôi nhập viện tại Bệnh viện Bạch Mai ngày 2024-01-15 vì viêm phổi.",
            "Tôi bị tai nạn xe máy ngày 2024-02-10, gãy tay phải.",
            "Hồ sơ của tôi đã được xử lý chưa?",
        ]
        transcript = random.choice(samples)
        latency = time.time() - t0
        logger.debug(f"[ASR] '{transcript}' ({latency:.3f}s)")
        return transcript, latency

    def generate_response(self, messages: list, audio_path: str = None) -> tuple[str, float]:
        """
        Giả lập model inference.
        Returns (raw_response_with_think, latency_s)
        """
        import random
        self._call_count += 1
        t0 = time.time()
        delay = random.uniform(*self.latency_range)
        time.sleep(delay)

        # Giả lập các loại response khác nhau
        scenario = self._call_count % 4
        if scenario == 0:
            raw = """<think>
Người dùng đang yêu cầu xác thực hợp đồng. Tôi cần gọi verify_policy với policy_id và customer_name.
Tôi thấy trong tin nhắn có đề cập HD113734 và Nguyễn Thị Mai.
</think>
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
}"""
        elif scenario == 1:
            raw = """<think>
Sau khi xác thực thành công, người dùng muốn yêu cầu bồi thường y tế.
Cần gọi process_medical_claim với thông tin bệnh viện và ngày nhập viện.
</think>
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_call",
      "name": "process_medical_claim",
      "arguments": {
        "policy_id": "HD113734",
        "hospital_name": "Bệnh viện Bạch Mai",
        "admission_date": "2024-01-15",
        "diagnosis": "Viêm phổi"
      }
    }
  ]
}"""
        elif scenario == 2:
            raw = """<think>
Đây là câu hỏi thông thường không cần tool. Người dùng hỏi về trạng thái hồ sơ.
Tôi sẽ trả lời trực tiếp với thông tin đã có.
</think>
Xin chào! Hồ sơ bồi thường của bạn đang được xem xét. 
Thời gian xử lý dự kiến là 5-7 ngày làm việc. 
Bạn có thể theo dõi qua mã ticket MED-3734."""
        else:
            raw = """<think>
Tình huống leo thang - người dùng không đồng ý với quyết định bồi thường.
Cần escalate lên manager.
</think>
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_call",
      "name": "escalate_claim_discrepancy",
      "arguments": {
        "claim_id": "MED-3734",
        "issue_type": "payout_discrepancy",
        "request_manager": true
      }
    }
  ]
}"""
        latency = time.time() - t0
        return raw, latency

    def text_to_speech(self, text: str, output_path: str) -> tuple[str, float]:
        """Giả lập TTS. Tạo file WAV trống."""
        import numpy as np
        t0 = time.time()
        time.sleep(0.1)
        # Tạo audio giả (1 giây silence)
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)
        import soundfile as sf
        sf.write(output_path, audio, sr)
        latency = time.time() - t0
        return output_path, latency


# ─────────────────────────────────────────────
# REAL MODEL CLIENT (Qwen2.5-Omni via transformers)
# ─────────────────────────────────────────────

class QwenOmniClient:
    """
    Real client cho Qwen2.5-7B-Omni.
    Load từ HuggingFace: thaomike/qwen-finetune-research
    """
    def __init__(self, model_name="thaomike/qwen-finetune-research", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        logger.info(f"[QwenOmni] Loading model: {self.model_name}")
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
            logger.info(f"[QwenOmni] Model loaded in {time.time()-t0:.1f}s")
        except ImportError:
            logger.error("[QwenOmni] transformers not installed or incompatible version")
            raise
        except Exception as e:
            logger.error(f"[QwenOmni] Load failed: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> tuple[str, float]:
        """ASR + audio encode via Qwen2.5-Omni processor."""
        import soundfile as sf
        t0 = time.time()
        audio, sr = sf.read(audio_path)
        # Build message với audio
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Transcribe the following audio accurately in Vietnamese."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "Transcribe this audio."}
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        import torch
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=256)
        transcript = self.processor.decode(out[0], skip_special_tokens=True)
        latency = time.time() - t0
        return transcript, latency

    def generate_response(self, messages: list, audio_path: str = None) -> tuple[str, float]:
        """
        Inference với conversation history.
        Messages format: [{"role": "user/assistant/system", "content": "..."}]
        """
        t0 = time.time()
        # Thêm system prompt với tool descriptions
        full_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ] + messages

        if audio_path:
            # Thêm audio vào message cuối
            last = full_messages[-1]
            if isinstance(last["content"], str):
                last["content"] = [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": last["content"]}
                ]

        inputs = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        import torch
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
            )
        raw = self.processor.decode(out[0], skip_special_tokens=True)
        latency = time.time() - t0
        return raw, latency

    def text_to_speech(self, text: str, output_path: str) -> tuple[str, float]:
        """
        TTS via Qwen2.5-Omni speech output.
        Nếu model hỗ trợ audio generation.
        """
        t0 = time.time()
        try:
            # Qwen2.5-Omni có thể generate audio trực tiếp
            messages = [
                {"role": "user", "content": [{"type": "text", "text": f"Đọc to câu sau bằng tiếng Việt: {text}"}]}
            ]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                return_tensors="pt", return_audio=True
            ).to(self.device)
            import torch
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=2048)
            # Extract audio từ output nếu có
            if hasattr(out, "audio"):
                import soundfile as sf
                sf.write(output_path, out.audio.cpu().numpy(), 24000)
            else:
                # Fallback: tạo file trống
                import numpy as np
                import soundfile as sf
                sf.write(output_path, np.zeros(16000, dtype=np.float32), 16000)
        except Exception as e:
            logger.warning(f"[TTS] Error: {e}. Creating silent audio.")
            import numpy as np
            import soundfile as sf
            sf.write(output_path, np.zeros(16000, dtype=np.float32), 16000)
        latency = time.time() - t0
        return output_path, latency