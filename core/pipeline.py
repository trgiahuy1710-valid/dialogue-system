"""
pipeline.py
───────────
Orchestrator chính:
  1. Load audio chunks từ user_x.wav
  2. Stream đến model (ASR → LLM → Tool → TTS)
  3. Ghi log latency chi tiết
  4. Emit events để Gradio UI cập nhật real-time
"""

import os
import re
import json
import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from pathlib import Path
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
    SYSTEM_PROMPT,
)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logging(log_dir: str, level=logging.DEBUG):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{ts}.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
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
    folders = [f for f in os.listdir(root) if f.startswith("gen-")]
    folders = sorted(folders, key=lambda x: int(re.findall(r'\d+', x)[0]))
    result = []
    for f in folders:
        num = int(re.findall(r'\d+', f)[0])
        if num >= start and (end is None or num <= end):
            result.append(f)
    return result


def get_user_files(folder: str):
    files = [f for f in os.listdir(folder) if f.startswith("user_") and f.endswith(".wav")]
    files = sorted(files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    return files


def get_audio_duration(path: str) -> float:
    try:
        audio, sr = sf.read(path)
        return len(audio) / sr
    except:
        return 0.0


# ─────────────────────────────────────────────
# EVENT SYSTEM
# ─────────────────────────────────────────────

@dataclass
class PipelineEvent:
    """Event được emit để UI cập nhật."""
    type: str  # "turn_start" | "asr_done" | "llm_done" | "tool_call" | "tool_result" | "tts_done" | "turn_end" | "error"
    conversation_id: str = ""
    turn_id: str = ""
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# CONVERSATION HISTORY
# ─────────────────────────────────────────────

class ConversationHistory:
    def __init__(self):
        self.messages = []

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_result(self, tool_name: str, result: str):
        self.messages.append({
            "role": "tool",
            "name": tool_name,
            "content": result
        })

    def get_messages(self):
        return self.messages.copy()


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

        # Setup logging
        self.log_file = setup_logging(log_dir)
        logger.info(f"Pipeline init | root={root_dir} | mock={use_mock}")

        # Load model
        if use_mock:
            self.model = MockModelClient()
        else:
            from core.model_client import QwenOmniClient
            self.model = QwenOmniClient(model_name)

        # Output dir cho audio responses
        self.output_audio_dir = os.path.join(log_dir, "response_audio")
        os.makedirs(self.output_audio_dir, exist_ok=True)

    def stop(self):
        self._stop_flag.set()

    def _emit(self, event: PipelineEvent):
        logger.debug(f"[EVENT] {event.type} | {event.payload}")
        try:
            self.on_event(event)
        except Exception as e:
            logger.warning(f"[EVENT] Handler error: {e}")

    def _stream_audio_chunks(self, audio_path: str) -> Generator[np.ndarray, None, None]:
        """Stream audio theo chunks."""
        audio, sr = sf.read(audio_path)
        chunk_size = int(sr * self.chunk_ms / 1000)
        for i in range(0, len(audio), chunk_size):
            if self._stop_flag.is_set():
                break
            chunk = audio[i:i+chunk_size]
            if self.realtime:
                time.sleep(self.chunk_ms / 1000)
            yield chunk

    def _process_turn(
        self,
        conv_id: str,
        turn_num: int,
        audio_path: str,
        history: ConversationHistory,
    ) -> TurnMetrics:
        """
        Xử lý một turn: ASR → LLM → (Tool?) → TTS
        """
        turn_id = f"{conv_id}_t{turn_num:02d}"
        metrics = TurnMetrics(turn_id=turn_id, audio_file=os.path.basename(audio_path))
        t_total_start = time.time()

        self._emit(PipelineEvent(
            type="turn_start",
            conversation_id=conv_id,
            turn_id=turn_id,
            payload={"audio_file": metrics.audio_file, "turn_num": turn_num}
        ))

        try:
            # ── 1. Audio duration
            metrics.audio_duration_s = get_audio_duration(audio_path)
            logger.info(f"[{turn_id}] Audio: {metrics.audio_file} ({metrics.audio_duration_s:.2f}s)")

            # ── 2. Stream audio (giả lập realtime đọc chunks)
            logger.debug(f"[{turn_id}] Streaming audio chunks...")
            chunk_count = 0
            for _ in self._stream_audio_chunks(audio_path):
                chunk_count += 1
            logger.debug(f"[{turn_id}] Streamed {chunk_count} chunks")

            # ── 3. ASR / Audio encode
            logger.info(f"[{turn_id}] → ASR...")
            t_asr = time.time()
            transcript, asr_lat = self.model.transcribe_audio(audio_path)
            metrics.audio_encode_latency_s = asr_lat
            logger.info(f"[{turn_id}] ASR: '{transcript}' ({asr_lat:.3f}s)")

            self._emit(PipelineEvent(
                type="asr_done",
                conversation_id=conv_id,
                turn_id=turn_id,
                payload={"transcript": transcript, "latency_s": round(asr_lat, 3)}
            ))

            # ── 4. LLM Inference
            history.add_user(transcript)
            logger.info(f"[{turn_id}] → LLM inference...")
            t_llm = time.time()
            raw_response, llm_lat = self.model.generate_response(
                history.get_messages(),
                audio_path=audio_path
            )
            metrics.inference_latency_s = llm_lat
            logger.info(f"[{turn_id}] LLM done ({llm_lat:.3f}s)")
            logger.debug(f"[{turn_id}] RAW response:\n{raw_response}")

            # ── 5. Extract thinker
            thinker, clean_response = extract_thinker(raw_response)
            metrics.thinker_text = thinker
            if thinker:
                logger.info(f"[{turn_id}] 🧠 Thinker:\n{thinker}")

            self._emit(PipelineEvent(
                type="llm_done",
                conversation_id=conv_id,
                turn_id=turn_id,
                payload={
                    "raw": raw_response,
                    "thinker": thinker,
                    "clean": clean_response,
                    "latency_s": round(llm_lat, 3)
                }
            ))

            # ── 6. Tool calls?
            tool_calls = parse_tool_call(clean_response)
            final_response = clean_response

            if tool_calls:
                logger.info(f"[{turn_id}] 🔧 Tool calls: {[tc['name'] for tc in tool_calls]}")
                self._emit(PipelineEvent(
                    type="tool_call",
                    conversation_id=conv_id,
                    turn_id=turn_id,
                    payload={"calls": tool_calls}
                ))

                t_tool = time.time()
                tool_results = execute_tools(tool_calls)
                metrics.tool_latency_s = time.time() - t_tool
                metrics.tool_calls = [
                    {"name": tc["name"], "args": tc.get("arguments", {})}
                    for tc in tool_calls
                ]

                # Ghi kết quả tool vào history
                for tr in tool_results:
                    result_str = tr.get("result", tr.get("error", ""))
                    history.add_tool_result(tr["tool"], result_str)
                    logger.info(f"[{turn_id}] Tool {tr['tool']} result: {result_str[:100]}")

                self._emit(PipelineEvent(
                    type="tool_result",
                    conversation_id=conv_id,
                    turn_id=turn_id,
                    payload={"results": tool_results, "latency_s": round(metrics.tool_latency_s, 3)}
                ))

                # LLM inference lần 2 với tool results
                logger.info(f"[{turn_id}] → LLM inference (post-tool)...")
                t_llm2 = time.time()
                raw2, llm2_lat = self.model.generate_response(history.get_messages())
                metrics.inference_latency_s += llm2_lat
                thinker2, final_response = extract_thinker(raw2)
                if thinker2:
                    metrics.thinker_text += f"\n[Post-tool]\n{thinker2}"
                logger.info(f"[{turn_id}] Post-tool LLM done ({llm2_lat:.3f}s): '{final_response[:80]}'")

            metrics.response_text = final_response
            history.add_assistant(final_response)

            # ── 7. TTS
            logger.info(f"[{turn_id}] → TTS...")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_audio = os.path.join(self.output_audio_dir, f"{turn_id}_{ts}.wav")
            _, tts_lat = self.model.text_to_speech(final_response, out_audio)
            metrics.tts_latency_s = tts_lat
            metrics.response_audio_path = out_audio
            logger.info(f"[{turn_id}] TTS done ({tts_lat:.3f}s) → {out_audio}")

            self._emit(PipelineEvent(
                type="tts_done",
                conversation_id=conv_id,
                turn_id=turn_id,
                payload={
                    "audio_path": out_audio,
                    "latency_s": round(tts_lat, 3),
                    "response_text": final_response[:200]
                }
            ))

        except Exception as e:
            metrics.error = str(e)
            logger.error(f"[{turn_id}] ERROR: {e}", exc_info=True)
            self._emit(PipelineEvent(
                type="error",
                conversation_id=conv_id,
                turn_id=turn_id,
                payload={"error": str(e)}
            ))

        metrics.total_latency_s = time.time() - t_total_start

        # ── 8. Log summary
        logger.info(
            f"[{turn_id}] ✅ SUMMARY | "
            f"audio={metrics.audio_duration_s:.2f}s | "
            f"asr={metrics.audio_encode_latency_s:.3f}s | "
            f"llm={metrics.inference_latency_s:.3f}s | "
            f"tool={metrics.tool_latency_s:.3f}s | "
            f"tts={metrics.tts_latency_s:.3f}s | "
            f"total={metrics.total_latency_s:.3f}s"
        )

        self._emit(PipelineEvent(
            type="turn_end",
            conversation_id=conv_id,
            turn_id=turn_id,
            payload=metrics.to_dict()
        ))

        self._metrics_history.append(metrics)
        return metrics

    def run_conversation(
        self,
        folder_name: str,
        max_files: Optional[int] = None,
    ) -> list[TurnMetrics]:
        """Chạy toàn bộ conversation trong một gen-xxxx folder."""
        full_path = os.path.join(self.root_dir, folder_name)
        user_files = get_user_files(full_path)
        if max_files:
            user_files = user_files[:max_files]

        conv_id = folder_name
        history = ConversationHistory()

        logger.info(f"\n{'='*60}")
        logger.info(f"📁 Conversation: {conv_id} ({len(user_files)} turns)")
        logger.info(f"{'='*60}")

        self._emit(PipelineEvent(
            type="conversation_start",
            conversation_id=conv_id,
            payload={"folder": folder_name, "total_turns": len(user_files)}
        ))

        results = []
        for i, fname in enumerate(user_files):
            if self._stop_flag.is_set():
                logger.info(f"[{conv_id}] Stopped by user")
                break
            audio_path = os.path.join(full_path, fname)
            metrics = self._process_turn(conv_id, i, audio_path, history)
            results.append(metrics)

        # Save metrics JSON
        self._save_metrics(conv_id, results)

        self._emit(PipelineEvent(
            type="conversation_end",
            conversation_id=conv_id,
            payload={"total_turns": len(results)}
        ))

        return results

    def run_all(
        self,
        start: int = 0,
        end: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> list[TurnMetrics]:
        """Chạy tất cả conversations."""
        folders = get_gen_folders(self.root_dir, start, end)
        logger.info(f"Running {len(folders)} conversations: {folders}")
        all_metrics = []
        for folder in folders:
            if self._stop_flag.is_set():
                break
            metrics = self.run_conversation(folder, max_files)
            all_metrics.extend(metrics)
        return all_metrics

    def _save_metrics(self, conv_id: str, metrics: list[TurnMetrics]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"metrics_{conv_id}_{ts}.json")
        data = {
            "conversation_id": conv_id,
            "timestamp": ts,
            "turns": [m.to_dict() for m in metrics],
            "summary": self._compute_summary(metrics)
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[Metrics] Saved → {path}")
        return path

    def _compute_summary(self, metrics: list[TurnMetrics]) -> dict:
        if not metrics:
            return {}
        
        def avg(vals): return round(sum(vals)/len(vals), 3) if vals else 0

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

    def get_all_metrics(self) -> list[dict]:
        return [m.to_dict() for m in self._metrics_history]