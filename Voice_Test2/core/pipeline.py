"""
pipeline.py  —  fixed
──────────────────────
Bug fixes & improvements:
  1. post-tool LLM: nếu response vẫn là tool_call JSON → extract text thuần, KHÔNG đưa vào TTS
  2. Thêm debug log cho từng bước folder scan, file list, audio path
  3. run_all: log rõ folders được chọn và lý do skip
  4. _process_turn: log rõ audio_path có tồn tại hay không trước khi xử lý
  5. ConversationHistory: log mỗi lần add message
"""

import os
import re
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Generator
from datetime import datetime

import soundfile as sf
import numpy as np

from core.model_client import (
    MockModelClient,
    TurnMetrics,
    extract_thinker,
    parse_tool_call,
    execute_tools,
    is_tool_call_json,
    SYSTEM_PROMPT,
)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def setup_logging(log_dir: str, level=logging.DEBUG):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{ts}.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    # Avoid duplicate handlers if init called multiple times
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
               for h in root.handlers):
        root.setLevel(level)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Console handler — add only once
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        root.addHandler(ch)

    return log_file


logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────
# FOLDER UTILS
# ─────────────────────────────────────────────

def get_gen_folders(root: str, start: int = 0, end: Optional[int] = None):
    if not os.path.isdir(root):
        logger.error(f"[Folders] Root directory does not exist: {root}")
        return []

    all_items = os.listdir(root)
    all_gen = [f for f in all_items if f.startswith("gen-")]
    all_gen = sorted(all_gen, key=lambda x: int(re.findall(r"\d+", x)[0]))

    logger.debug(f"[Folders] Root: {root}")
    logger.debug(f"[Folders] All items: {all_items}")
    logger.debug(f"[Folders] gen- folders found: {all_gen}")

    result = []
    for f in all_gen:
        num = int(re.findall(r"\d+", f)[0])
        if num >= start and (end is None or num <= end):
            result.append(f)
        else:
            logger.debug(f"[Folders] Skip {f} (num={num}, range=[{start}, {end}])")

    logger.info(f"[Folders] Selected {len(result)}/{len(all_gen)} folders: {result}")
    return result


def get_user_files(folder: str):
    if not os.path.isdir(folder):
        logger.error(f"[Files] Folder not found: {folder}")
        return []

    all_files = os.listdir(folder)
    user_files = [f for f in all_files if f.startswith("user_") and f.endswith(".wav")]
    user_files = sorted(user_files, key=lambda x: int(re.findall(r"\d+", x)[0]))

    logger.debug(f"[Files] Folder: {folder}")
    logger.debug(f"[Files] All files: {all_files}")
    logger.debug(f"[Files] user_*.wav found: {user_files}")

    if not user_files:
        logger.warning(f"[Files] No user_*.wav found in {folder}!")

    return user_files


def get_audio_duration(path: str) -> float:
    try:
        audio, sr = sf.read(path)
        return len(audio) / sr
    except Exception as e:
        logger.error(f"[Audio] Cannot read duration from {path}: {e}")
        return 0.0


# ─────────────────────────────────────────────
# EVENTS
# ─────────────────────────────────────────────

@dataclass
class PipelineEvent:
    type: str
    conversation_id: str = ""
    turn_id: str = ""
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# CONVERSATION HISTORY
# ─────────────────────────────────────────────

class ConversationHistory:
    def __init__(self, conv_id: str = ""):
        self.messages = []
        self.conv_id = conv_id

    def add_user(self, text: str):
        logger.debug(f"[History:{self.conv_id}] add_user: '{text[:80]}'")
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        logger.debug(f"[History:{self.conv_id}] add_assistant: '{text[:80]}'")
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_result(self, tool_name: str, result: str):
        logger.debug(f"[History:{self.conv_id}] add_tool_result: {tool_name} → '{result[:80]}'")
        self.messages.append({"role": "tool", "name": tool_name, "content": result})

    def get_messages(self):
        return self.messages.copy()

    def summary(self) -> str:
        return f"{len(self.messages)} msgs: " + " | ".join(
            f"{m['role']}:{str(m.get('content',''))[:30]}" for m in self.messages
        )


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

class AudioTestPipeline:
    def __init__(
        self,
        root_dir: str,
        log_dir: str = "./logs",
        use_mock: bool = True,
        model_name: str = "thaomike/qwen-finetune-research",
        chunk_ms: int = 20,
        realtime: bool = False,
        on_event: Optional[Callable[[PipelineEvent], None]] = None,
    ):
        self.root_dir = root_dir
        self.log_dir = log_dir
        self.chunk_ms = chunk_ms
        self.realtime = realtime
        self.on_event = on_event or (lambda e: None)
        self._stop_flag = threading.Event()
        self._metrics_history: list[TurnMetrics] = []

        self.log_file = setup_logging(log_dir)
        logger.info(f"[Pipeline] Init | root_dir={root_dir} | mock={use_mock} | chunk_ms={chunk_ms}")
        logger.info(f"[Pipeline] root_dir exists: {os.path.isdir(root_dir)}")

        # ── Load model ──
        if use_mock:
            self.model = MockModelClient()
            logger.info("[Pipeline] Using MockModelClient")
        else:
            from core.model_client import QwenOmniClient
            logger.info(f"[Pipeline] Loading QwenOmniClient: {model_name}")
            self.model = QwenOmniClient(model_name)

        self.output_audio_dir = os.path.join(log_dir, "response_audio")
        os.makedirs(self.output_audio_dir, exist_ok=True)
        logger.info(f"[Pipeline] Response audio dir: {self.output_audio_dir}")

    def stop(self):
        self._stop_flag.set()
        logger.info("[Pipeline] Stop flag set")

    def _emit(self, event: PipelineEvent):
        logger.debug(f"[Event] {event.type} | conv={event.conversation_id} | turn={event.turn_id}")
        try:
            self.on_event(event)
        except Exception as e:
            logger.warning(f"[Event] Handler exception: {e}")

    def _stream_audio_chunks(self, audio_path: str) -> Generator[np.ndarray, None, None]:
        audio, sr = sf.read(audio_path)
        chunk_size = int(sr * self.chunk_ms / 1000)
        total_chunks = len(range(0, len(audio), chunk_size))
        logger.debug(f"[Stream] {os.path.basename(audio_path)} | {total_chunks} chunks × {self.chunk_ms}ms")
        for i in range(0, len(audio), chunk_size):
            if self._stop_flag.is_set():
                break
            if self.realtime:
                time.sleep(self.chunk_ms / 1000)
            yield audio[i: i + chunk_size]

    # ── BUG FIX: extract clean text từ post-tool response ──
    def _extract_final_text(self, raw: str, turn_id: str) -> str:
        """
        Trả về text thuần để TTS đọc.
        Nếu response vẫn là tool_call JSON → trả về thông báo thay thế.
        Nếu có <think> → strip ra.
        """
        _, clean = extract_thinker(raw)

        if is_tool_call_json(clean.strip()):
            logger.error(
                f"[{turn_id}] BUG: post-tool LLM still returned tool_call JSON → using fallback text\n"
                f"  Preview: {clean[:200]}"
            )
            return "Yêu cầu của bạn đã được ghi nhận và đang được xử lý. Chúng tôi sẽ liên hệ lại sớm."

        if not clean.strip():
            logger.warning(f"[{turn_id}] Empty response after thinker strip → using fallback")
            return "Xin lỗi, tôi không có phản hồi phù hợp lúc này."

        return clean

    def _process_turn(
        self,
        conv_id: str,
        turn_num: int,
        audio_path: str,
        history: ConversationHistory,
    ) -> TurnMetrics:
        turn_id = f"{conv_id}_t{turn_num:02d}"
        metrics = TurnMetrics(turn_id=turn_id, audio_file=os.path.basename(audio_path))
        t_total = time.time()

        # ── Verify file trước khi bắt đầu ──
        file_exists = os.path.exists(audio_path)
        logger.info(f"[{turn_id}] ─────────────────────────────────────")
        logger.info(f"[{turn_id}] START TURN {turn_num} | file={audio_path}")
        logger.info(f"[{turn_id}] File exists: {file_exists}")
        if not file_exists:
            logger.error(f"[{turn_id}] AUDIO FILE NOT FOUND — skipping turn")
            metrics.error = f"File not found: {audio_path}"
            self._emit(PipelineEvent(
                type="error", conversation_id=conv_id, turn_id=turn_id,
                payload={"error": metrics.error}
            ))
            metrics.total_latency_s = time.time() - t_total
            return metrics

        self._emit(PipelineEvent(
            type="turn_start", conversation_id=conv_id, turn_id=turn_id,
            payload={"audio_file": metrics.audio_file, "turn_num": turn_num}
        ))

        try:
            # 1. Duration
            metrics.audio_duration_s = get_audio_duration(audio_path)
            logger.info(f"[{turn_id}] Duration: {metrics.audio_duration_s:.2f}s")

            # 2. Stream chunks
            chunk_count = sum(1 for _ in self._stream_audio_chunks(audio_path))
            logger.debug(f"[{turn_id}] Streamed {chunk_count} chunks")

            # 3. ASR
            logger.info(f"[{turn_id}] → ASR")
            transcript, asr_lat = self.model.transcribe_audio(audio_path)
            metrics.audio_encode_latency_s = asr_lat
            logger.info(f"[{turn_id}] ASR [{asr_lat:.3f}s]: '{transcript}'")
            self._emit(PipelineEvent(
                type="asr_done", conversation_id=conv_id, turn_id=turn_id,
                payload={"transcript": transcript, "latency_s": round(asr_lat, 3)}
            ))

            # 4. LLM (turn 1)
            history.add_user(transcript)
            logger.info(f"[{turn_id}] → LLM | history: {history.summary()}")
            raw, llm_lat = self.model.generate_response(history.get_messages(), audio_path=audio_path)
            metrics.inference_latency_s = llm_lat
            logger.debug(f"[{turn_id}] LLM RAW ({llm_lat:.3f}s):\n{raw}")

            thinker, clean = extract_thinker(raw)
            metrics.thinker_text = thinker
            if thinker:
                logger.info(f"[{turn_id}] 🧠 Thinker: {thinker[:200]}")
            self._emit(PipelineEvent(
                type="llm_done", conversation_id=conv_id, turn_id=turn_id,
                payload={"raw": raw, "thinker": thinker, "clean": clean, "latency_s": round(llm_lat, 3)}
            ))

            # 5. Tool calls?
            tool_calls = parse_tool_call(clean)
            final_text = clean

            if tool_calls:
                logger.info(f"[{turn_id}] 🔧 Tools detected: {[tc['name'] for tc in tool_calls]}")
                self._emit(PipelineEvent(
                    type="tool_call", conversation_id=conv_id, turn_id=turn_id,
                    payload={"calls": tool_calls}
                ))

                t_tool = time.time()
                tool_results = execute_tools(tool_calls)
                metrics.tool_latency_s = time.time() - t_tool
                metrics.tool_calls = [
                    {"name": tc["name"], "args": tc.get("arguments", {})} for tc in tool_calls
                ]

                for tr in tool_results:
                    history.add_tool_result(tr["tool"], str(tr.get("result", tr.get("error", ""))))

                self._emit(PipelineEvent(
                    type="tool_result", conversation_id=conv_id, turn_id=turn_id,
                    payload={"results": tool_results, "latency_s": round(metrics.tool_latency_s, 3)}
                ))

                # LLM turn 2 (post-tool)
                logger.info(f"[{turn_id}] → LLM (post-tool) | history: {history.summary()}")
                raw2, llm2_lat = self.model.generate_response(history.get_messages())
                metrics.inference_latency_s += llm2_lat
                logger.debug(f"[{turn_id}] Post-tool RAW ({llm2_lat:.3f}s):\n{raw2}")

                thinker2, clean2 = extract_thinker(raw2)
                if thinker2:
                    metrics.thinker_text += f"\n[Post-tool]\n{thinker2}"

                # ── BUG FIX: guard final_text ──
                final_text = self._extract_final_text(raw2, turn_id)
                logger.info(f"[{turn_id}] Final text for TTS: '{final_text[:120]}'")

            else:
                logger.info(f"[{turn_id}] No tool calls — direct response")
                final_text = self._extract_final_text(raw, turn_id)

            metrics.response_text = final_text
            history.add_assistant(final_text)

            # 6. TTS
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_audio = os.path.join(self.output_audio_dir, f"{turn_id}_{ts}.wav")
            logger.info(f"[{turn_id}] → TTS: '{final_text[:80]}'")
            _, tts_lat = self.model.text_to_speech(final_text, out_audio)
            metrics.tts_latency_s = tts_lat
            metrics.response_audio_path = out_audio
            logger.info(f"[{turn_id}] TTS [{tts_lat:.3f}s] → {out_audio}")
            self._emit(PipelineEvent(
                type="tts_done", conversation_id=conv_id, turn_id=turn_id,
                payload={"audio_path": out_audio, "latency_s": round(tts_lat, 3),
                         "response_text": final_text[:200]}
            ))

        except Exception as e:
            metrics.error = str(e)
            logger.error(f"[{turn_id}] EXCEPTION: {e}", exc_info=True)
            self._emit(PipelineEvent(
                type="error", conversation_id=conv_id, turn_id=turn_id,
                payload={"error": str(e)}
            ))

        metrics.total_latency_s = time.time() - t_total
        logger.info(
            f"[{turn_id}] ✅ SUMMARY "
            f"audio={metrics.audio_duration_s:.2f}s | "
            f"asr={metrics.audio_encode_latency_s:.3f}s | "
            f"llm={metrics.inference_latency_s:.3f}s | "
            f"tool={metrics.tool_latency_s:.3f}s | "
            f"tts={metrics.tts_latency_s:.3f}s | "
            f"total={metrics.total_latency_s:.3f}s"
        )
        self._emit(PipelineEvent(
            type="turn_end", conversation_id=conv_id, turn_id=turn_id,
            payload=metrics.to_dict()
        ))
        self._metrics_history.append(metrics)
        return metrics

    def run_conversation(self, folder_name: str, max_files: Optional[int] = None) -> list[TurnMetrics]:
        full_path = os.path.join(self.root_dir, folder_name)
        logger.info(f"\n{'='*60}")
        logger.info(f"[Conv] START: {folder_name} | full_path={full_path}")
        logger.info(f"[Conv] Path exists: {os.path.isdir(full_path)}")

        user_files = get_user_files(full_path)
        if not user_files:
            logger.error(f"[Conv] No user files in {full_path} — skipping conversation")
            return []

        if max_files:
            user_files = user_files[:max_files]
            logger.info(f"[Conv] Limited to {max_files} files: {user_files}")

        history = ConversationHistory(conv_id=folder_name)

        self._emit(PipelineEvent(
            type="conversation_start", conversation_id=folder_name,
            payload={"folder": folder_name, "total_turns": len(user_files),
                     "files": user_files}
        ))

        results = []
        for i, fname in enumerate(user_files):
            if self._stop_flag.is_set():
                logger.info(f"[Conv] Stopped by user after {i} turns")
                break
            audio_path = os.path.join(full_path, fname)
            logger.info(f"[Conv] Processing turn {i+1}/{len(user_files)}: {fname}")
            metrics = self._process_turn(folder_name, i, audio_path, history)
            results.append(metrics)

        self._save_metrics(folder_name, results)
        self._emit(PipelineEvent(
            type="conversation_end", conversation_id=folder_name,
            payload={"total_turns": len(results)}
        ))
        logger.info(f"[Conv] DONE: {folder_name} | {len(results)} turns")
        return results

    def run_all(self, start: int = 0, end: Optional[int] = None, max_files: Optional[int] = None):
        logger.info(f"[RunAll] root={self.root_dir} | start={start} end={end} max_files={max_files}")
        folders = get_gen_folders(self.root_dir, start, end)

        if not folders:
            logger.error(f"[RunAll] No folders found! Check root_dir and start/end range.")
            return []

        logger.info(f"[RunAll] {len(folders)} folders to process: {folders}")
        all_metrics = []
        for i, folder in enumerate(folders):
            if self._stop_flag.is_set():
                logger.info(f"[RunAll] Stopped at folder {i}/{len(folders)}")
                break
            logger.info(f"[RunAll] [{i+1}/{len(folders)}] → {folder}")
            metrics = self.run_conversation(folder, max_files)
            all_metrics.extend(metrics)

        logger.info(f"[RunAll] COMPLETE | {len(all_metrics)} total turns")
        return all_metrics

    def _save_metrics(self, conv_id: str, metrics: list[TurnMetrics]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"metrics_{conv_id}_{ts}.json")
        data = {
            "conversation_id": conv_id,
            "timestamp": ts,
            "turns": [m.to_dict() for m in metrics],
            "summary": self._compute_summary(metrics),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[Metrics] Saved → {path}")

    def _compute_summary(self, metrics: list[TurnMetrics]) -> dict:
        if not metrics:
            return {}
        def avg(vals): return round(sum(vals) / len(vals), 3) if vals else 0
        return {
            "total_turns": len(metrics),
            "avg_asr_latency_s": avg([m.audio_encode_latency_s for m in metrics]),
            "avg_llm_latency_s": avg([m.inference_latency_s for m in metrics]),
            "avg_tool_latency_s": avg([m.tool_latency_s for m in metrics]),
            "avg_tts_latency_s": avg([m.tts_latency_s for m in metrics]),
            "avg_total_latency_s": avg([m.total_latency_s for m in metrics]),
            "turns_with_tools": sum(1 for m in metrics if m.tool_calls),
            "turns_with_errors": sum(1 for m in metrics if m.error),
        }

    def get_all_metrics(self):
        return [m.to_dict() for m in self._metrics_history]