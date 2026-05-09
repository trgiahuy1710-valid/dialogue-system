"""
app.py  —  fixed (Gradio 6.x)
──────────────────────────────
Bug fixes:
  1. css/theme → launch() thay vì Blocks()
  2. Bỏ buttons=["copy"] khỏi Textbox (gây lỗi ở Gradio < 6.x trên server)
  3. Bỏ max_lines từ Textbox (dùng lines thay thế)
  4. Thêm Tab "🐛 Debug" hiển thị trạng thái folder/file scan
  5. run_pipeline: log rõ root_dir, folders, files trước khi chạy
  6. log_lines dedup: build_refresh_payload dùng log_version counter
     thay vì so sánh list → tránh trùng lặp khi Timer và load cùng gọi
"""

import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from core.model_client import SYSTEM_PROMPT
from core.pipeline import AudioTestPipeline, PipelineEvent, get_gen_folders, get_user_files

logger = logging.getLogger("gradio_app")


# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────

pipeline_instance: AudioTestPipeline | None = None
run_thread: threading.Thread | None = None
current_metrics: list[dict] = []
log_lines: list[str] = []
last_audio_path_sent: str | None = None
state_lock = threading.Lock()

# Debug info: folder/file scan results
debug_lines: list[str] = []

MAX_LOG_LINES = 500


def reset_runtime_state() -> None:
    global current_metrics, log_lines, last_audio_path_sent, debug_lines
    with state_lock:
        current_metrics = []
        log_lines = []
        debug_lines = []
        last_audio_path_sent = None


def add_log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    icon = {"DEBUG": "·", "INFO": "►", "WARN": "⚠", "ERROR": "✖", "SUCCESS": "✔"}.get(level, "·")
    with state_lock:
        log_lines.append(f"[{ts}] {icon} {msg}")
        if len(log_lines) > MAX_LOG_LINES:
            log_lines.pop(0)


def add_debug(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with state_lock:
        debug_lines.append(f"[{ts}] {msg}")
        if len(debug_lines) > 500:
            debug_lines.pop(0)


def snapshot_state():
    with state_lock:
        return list(current_metrics), list(log_lines), list(debug_lines)


# ─────────────────────────────────────────────
# EVENT HANDLER
# ─────────────────────────────────────────────

def on_pipeline_event(event: PipelineEvent) -> None:
    etype = event.type
    p = event.payload
    tid = event.turn_id or event.conversation_id

    if etype == "conversation_start":
        files = p.get("files", [])
        add_log(f"📁 START CONVERSATION: {event.conversation_id} ({p.get('total_turns', 0)} turns)", "INFO")
        add_debug(f"[conv_start] {event.conversation_id} | files={files}")

    elif etype == "turn_start":
        add_log(f"┌─ TURN {p.get('turn_num', 0) + 1}: {p.get('audio_file', '')}", "INFO")
        add_debug(f"[turn_start] {tid} | audio_file={p.get('audio_file')}")

    elif etype == "asr_done":
        add_log(f"│  ASR [{p.get('latency_s', 0):.3f}s]: \"{p.get('transcript', '')}\"", "INFO")
        add_debug(f"[asr_done] {tid} | lat={p.get('latency_s', 0):.3f}s | transcript='{p.get('transcript','')}'")

    elif etype == "llm_done":
        thinker = p.get("thinker", "")
        lat = p.get("latency_s", 0)
        clean = p.get("clean", "")
        if thinker:
            add_log(f"│  🧠 THINKER [{lat:.3f}s]: {thinker[:120]}", "INFO")
        else:
            add_log(f"│  LLM [{lat:.3f}s]: {clean[:120]}", "INFO")
        add_debug(f"[llm_done] {tid} | lat={lat:.3f}s | has_think={bool(thinker)} | clean_preview='{clean[:80]}'")

    elif etype == "tool_call":
        for c in p.get("calls", []):
            args = json.dumps(c.get("arguments", {}), ensure_ascii=False)[:80]
            add_log(f"│  🔧 TOOL CALL: {c['name']}({args})", "WARN")
            add_debug(f"[tool_call] {tid} | tool={c['name']} args={args}")

    elif etype == "tool_result":
        for r in p.get("results", []):
            res = str(r.get("result", r.get("error", "")))[:100]
            add_log(f"│  ✔ TOOL RESULT [{p.get('latency_s', 0):.3f}s]: {res}", "SUCCESS")
            add_debug(f"[tool_result] {tid} | tool={r.get('tool')} | result='{res}'")

    elif etype == "tts_done":
        text = p.get("response_text", "")
        add_log(f"│  🔊 TTS [{p.get('latency_s', 0):.3f}s]: {text[:100]}", "INFO")
        add_debug(f"[tts_done] {tid} | lat={p.get('latency_s',0):.3f}s | audio={p.get('audio_path','')}")

    elif etype == "turn_end":
        add_log(
            f"└─ DONE | total={p.get('total_latency_s', 0):.3f}s "
            f"asr={p.get('audio_encode_latency_s', 0):.3f}s "
            f"llm={p.get('inference_latency_s', 0):.3f}s "
            f"tool={p.get('tool_latency_s', 0):.3f}s "
            f"tts={p.get('tts_latency_s', 0):.3f}s",
            "SUCCESS",
        )
        with state_lock:
            current_metrics.append(p)

    elif etype == "error":
        add_log(f"✖ ERROR [{tid}]: {p.get('error', '')}", "ERROR")
        add_debug(f"[error] {tid} | {p.get('error','')}")

    elif etype == "conversation_end":
        add_log(f"✔ CONVERSATION DONE: {event.conversation_id} ({p.get('total_turns', 0)} turns)", "SUCCESS")


# ─────────────────────────────────────────────
# PIPELINE CONTROL
# ─────────────────────────────────────────────

def init_pipeline(root_dir: str, use_mock: bool, model_name: str, log_dir: str):
    global pipeline_instance

    reset_runtime_state()

    add_debug(f"[init] root_dir='{root_dir}' | exists={os.path.isdir(root_dir)}")

    if not root_dir or not os.path.isdir(root_dir):
        msg = f"❌ Root directory không tồn tại: '{root_dir}'"
        add_log(msg, "ERROR")
        add_debug(f"[init] FAIL: {msg}")
        return msg, gr.update(interactive=False)

    # Debug: scan folders
    try:
        all_items = os.listdir(root_dir)
        gen_folders = [f for f in all_items if f.startswith("gen-")]
        add_debug(f"[init] listdir({root_dir}) = {all_items}")
        add_debug(f"[init] gen- folders = {gen_folders}")
    except Exception as e:
        add_debug(f"[init] listdir ERROR: {e}")

    try:
        pipeline_instance = AudioTestPipeline(
            root_dir=root_dir,
            log_dir=log_dir or "./logs",
            use_mock=bool(use_mock),
            model_name=model_name or "thaomike/qwen-finetune-research",
            on_event=on_pipeline_event,
        )
        folders = get_gen_folders(root_dir)
        add_log(f"Pipeline initialized | {len(folders)} folders | mock={use_mock}", "SUCCESS")
        add_debug(f"[init] OK | {len(folders)} conversations: {folders}")
        return f"✔ Pipeline ready | {len(folders)} conversations", gr.update(interactive=True)
    except Exception as exc:
        add_log(f"Init failed: {exc}", "ERROR")
        add_debug(f"[init] EXCEPTION: {exc}")
        return f"❌ Init failed: {exc}", gr.update(interactive=False)


def run_pipeline(root_dir, start_idx, end_idx, max_files, chunk_ms, realtime_mode, selected_folder):
    global run_thread

    if pipeline_instance is None:
        add_log("Pipeline not initialized!", "ERROR")
        return "❌ Not initialized"

    # Reset metrics but keep debug log
    with state_lock:
        current_metrics.clear()
        log_lines.clear()

    pipeline_instance._stop_flag.clear()
    pipeline_instance.root_dir = root_dir
    pipeline_instance.chunk_ms = int(chunk_ms or 20)
    pipeline_instance.realtime = bool(realtime_mode)

    # Debug: pre-run folder verification
    add_debug(f"[run] root_dir='{root_dir}' | exists={os.path.isdir(root_dir)}")
    if os.path.isdir(root_dir):
        folders = get_gen_folders(root_dir, int(start_idx or 0), int(end_idx) if end_idx else None)
        add_debug(f"[run] folders selected: {folders}")
        for f in folders[:5]:  # first 5
            fp = os.path.join(root_dir, f)
            ufiles = get_user_files(fp)
            add_debug(f"[run] {f} → user files: {ufiles}")

    def _run() -> None:
        try:
            if selected_folder and selected_folder != "ALL":
                add_log(f"Running single conversation: {selected_folder}", "INFO")
                add_debug(f"[run] mode=single | folder={selected_folder}")
                pipeline_instance.run_conversation(
                    selected_folder,
                    max_files=int(max_files) if max_files else None,
                )
            else:
                add_log(f"Running all conversations [{start_idx}→{end_idx}]", "INFO")
                add_debug(f"[run] mode=all | start={start_idx} end={end_idx} max_files={max_files}")
                pipeline_instance.run_all(
                    start=int(start_idx or 0),
                    end=int(end_idx) if end_idx else None,
                    max_files=int(max_files) if max_files else None,
                )
        except Exception as exc:
            add_log(f"Run error: {exc}", "ERROR")
            add_debug(f"[run] EXCEPTION: {exc}")

    run_thread = threading.Thread(target=_run, daemon=True)
    run_thread.start()
    return "▶ Running..."


def stop_pipeline():
    if pipeline_instance:
        pipeline_instance.stop()
        add_log("⚡ Stop requested", "WARN")
    return "⚡ Stopping..."


def get_folder_list(root_dir: str):
    if not root_dir or not os.path.isdir(root_dir):
        add_debug(f"[folder_list] invalid root: '{root_dir}'")
        return gr.update(choices=["ALL"], value="ALL")
    folders = get_gen_folders(root_dir)
    add_debug(f"[folder_list] found: {folders}")
    return gr.update(choices=["ALL"] + folders, value="ALL")


# ─────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────

def format_metrics_table(metrics: list[dict]) -> str:
    if not metrics:
        return "No data yet."
    lines = [
        f"{'TURN_ID':<28} {'AUDIO':>7} {'ASR':>7} {'LLM':>7} {'TOOL':>7} {'TTS':>7} {'TOTAL':>7}  TOOLS  ERR",
        "─" * 100,
    ]
    for m in metrics[-80:]:
        tid   = m.get("turn_id", "")[:27].ljust(27)
        aud   = f"{m.get('audio_duration_s', 0):.2f}s"
        asr   = f"{m.get('audio_encode_latency_s', 0):.3f}s"
        llm   = f"{m.get('inference_latency_s', 0):.3f}s"
        tool  = f"{m.get('tool_latency_s', 0):.3f}s"
        tts   = f"{m.get('tts_latency_s', 0):.3f}s"
        total = f"{m.get('total_latency_s', 0):.3f}s"
        tools = ", ".join(t["name"] for t in m.get("tool_calls", []))[:22].ljust(22)
        err   = "✖" if m.get("error") else "·"
        lines.append(f"{tid} {aud:>7} {asr:>7} {llm:>7} {tool:>7} {tts:>7} {total:>7}  {tools} {err}")

    n = len(metrics)
    lines.append("─" * 100)
    lines.append(
        f"{'AVG (' + str(n) + ' turns)':<28} "
        f"{'':>7} "
        f"{sum(m.get('audio_encode_latency_s', 0) for m in metrics)/n:>6.3f}s "
        f"{sum(m.get('inference_latency_s', 0) for m in metrics)/n:>6.3f}s "
        f"{sum(m.get('tool_latency_s', 0) for m in metrics)/n:>6.3f}s "
        f"{sum(m.get('tts_latency_s', 0) for m in metrics)/n:>6.3f}s "
        f"{sum(m.get('total_latency_s', 0) for m in metrics)/n:>6.3f}s"
    )
    return "\n".join(lines)


def format_thinker_log(metrics: list[dict]) -> str:
    if not metrics:
        return "No thinker data."
    out = []
    for m in metrics:
        t = m.get("thinker_text", "").strip()
        if t:
            out.append(f"── {m.get('turn_id', '')} ──────────────────────")
            out.append(t)
            out.append("")
    return "\n".join(out) if out else "No <think> blocks found yet."


def build_tool_history(metrics: list[dict]) -> list[dict]:
    h = []
    for m in metrics:
        for tc in m.get("tool_calls", []):
            h.append({"turn": m.get("turn_id"), "tool": tc.get("name"), "args": tc.get("args", {})})
    return h


def find_latest_audio(metrics: list[dict]) -> str | None:
    for m in reversed(metrics):
        p = m.get("response_audio_path")
        if p and os.path.exists(p):
            return p
    return None


def get_audio_update(latest_path):
    global last_audio_path_sent
    with state_lock:
        if latest_path == last_audio_path_sent:
            return gr.skip()
        last_audio_path_sent = latest_path
    return gr.update(value=latest_path)


def build_refresh_payload():
    metrics, logs, dbg = snapshot_state()

    n = len(metrics)
    avg_total = f"{sum(m.get('total_latency_s', 0) for m in metrics)/n:.3f}s" if n else "0.000s"
    avg_llm   = f"{sum(m.get('inference_latency_s', 0) for m in metrics)/n:.3f}s" if n else "0.000s"
    tools_n   = sum(len(m.get("tool_calls", [])) for m in metrics)
    latest    = find_latest_audio(metrics)

    return (
        "\n".join(logs[-120:]),
        format_metrics_table(metrics),
        format_thinker_log(metrics),
        build_tool_history(metrics),
        metrics[-50:] if metrics else [],
        str(n),
        avg_total,
        avg_llm,
        str(tools_n),
        get_audio_update(latest),
        "\n".join(dbg[-200:]),  # debug tab
    )


def manual_refresh_metrics():
    m, _, _ = snapshot_state()
    return format_metrics_table(m)


def manual_refresh_thinker():
    m, _, _ = snapshot_state()
    return format_thinker_log(m)


def manual_refresh_tools():
    m, _, _ = snapshot_state()
    return build_tool_history(m)


def manual_refresh_json():
    m, _, _ = snapshot_state()
    return m[-50:] if m else []


# ─────────────────────────────────────────────
# FONT & CSS
# ─────────────────────────────────────────────

FONT_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
"""

CSS = """
:root {
  --bg-0: #0a0c0f; --bg-1: #10131a; --bg-2: #161b26;
  --border: #2a3347; --accent-cyan: #00d4ff; --accent-green: #00ff88;
  --accent-red: #ff4444; --text-primary: #e2e8f0; --text-dim: #475569;
  --mono: 'JetBrains Mono', monospace; --sans: 'IBM Plex Sans', sans-serif;
}
body, .gradio-container { background: var(--bg-0) !important; font-family: var(--sans) !important; color: var(--text-primary) !important; }
.gradio-container { max-width: 1600px !important; }

.app-header { background: var(--bg-1); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
.app-title { font-family: var(--mono); font-size: 18px; font-weight: 700; color: var(--accent-cyan); letter-spacing: .05em; text-transform: uppercase; }
.app-subtitle { font-family: var(--mono); font-size: 11px; color: var(--text-dim); letter-spacing: .1em; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

.section-header { font-family: var(--mono); font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: .2em; padding: 4px 0; border-bottom: 1px solid var(--border); margin-bottom: 8px; }

input[type="text"], input[type="number"], select, textarea { background: var(--bg-2) !important; border: 1px solid var(--border) !important; color: var(--text-primary) !important; font-family: var(--mono) !important; font-size: 12px !important; border-radius: 3px !important; }
input:focus, textarea:focus { border-color: var(--accent-cyan) !important; outline: none !important; box-shadow: 0 0 0 1px var(--accent-cyan) !important; }
label, .label-wrap span { font-family: var(--mono) !important; font-size: 11px !important; color: var(--text-dim) !important; text-transform: uppercase !important; letter-spacing: .1em !important; }

button { font-family: var(--mono) !important; font-size: 12px !important; letter-spacing: .08em !important; text-transform: uppercase !important; border-radius: 3px !important; transition: all .15s !important; }
button.primary { background: transparent !important; border: 1px solid var(--accent-cyan) !important; color: var(--accent-cyan) !important; }
button.primary:hover { background: var(--accent-cyan) !important; color: var(--bg-0) !important; box-shadow: 0 0 12px rgba(0,212,255,.3) !important; }
button.secondary { background: transparent !important; border: 1px solid var(--border) !important; color: #94a3b8 !important; }
.stop-btn { border-color: var(--accent-red) !important; color: var(--accent-red) !important; }
.stop-btn:hover { background: var(--accent-red) !important; color: white !important; }

.log-terminal textarea { background: #050709 !important; color: #a0ffb0 !important; font-family: var(--mono) !important; font-size: 11px !important; line-height: 1.6 !important; border: 1px solid #1a2a1a !important; border-radius: 3px !important; }
.metrics-table textarea { background: var(--bg-1) !important; color: #94a3b8 !important; font-family: var(--mono) !important; font-size: 11px !important; line-height: 1.5 !important; }
.thinker-box textarea { background: #0f0a1a !important; color: #c084fc !important; font-family: var(--mono) !important; font-size: 11px !important; border-color: #3b1f6e !important; }
.debug-box textarea { background: #0a0a0a !important; color: #facc15 !important; font-family: var(--mono) !important; font-size: 10.5px !important; line-height: 1.5 !important; border-color: #3a3000 !important; }
audio { width: 100%; filter: invert(.85) hue-rotate(180deg); }
"""


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="VoiceTest System", head=FONT_HEAD) as demo:

        gr.HTML("""
        <div class="app-header">
          <div class="status-dot"></div>
          <div>
            <div class="app-title">◈ VoiceTest System v1.1</div>
            <div class="app-subtitle">Qwen2.5-7B-Omni · Insurance Claim Tester · Debug Mode</div>
          </div>
        </div>
        """)

        with gr.Row():
            # ── LEFT: Config ──────────────────────────────────────
            with gr.Column(scale=1, min_width=320):
                gr.HTML('<div class="section-header">// Pipeline Config</div>')

                root_dir = gr.Textbox(
                    label="Root Directory",
                    placeholder="/storage/huytg/dataset_audio_vieneu_clean_f5tts_v2",
                    value="./sample_data",
                )
                log_dir = gr.Textbox(label="Log Directory", value="./logs")
                model_name = gr.Textbox(
                    label="Model Name (HuggingFace)",
                    value="thaomike/qwen-finetune-research",
                )
                use_mock = gr.Checkbox(label="Use Mock Model (no GPU)", value=True)

                btn_init = gr.Button("⚡ INITIALIZE PIPELINE", variant="primary")
                init_status = gr.Textbox(label="Status", value="Not initialized", interactive=False, lines=1)

                gr.HTML('<div class="section-header">// Run Config</div>')

                folder_select = gr.Dropdown(label="Conversation (folder)", choices=["ALL"], value="ALL")

                with gr.Row():
                    start_idx = gr.Number(label="Start Index", value=0, minimum=0)
                    end_idx = gr.Number(label="End Index", value=None)

                with gr.Row():
                    max_files = gr.Number(label="Max Turns/Conv", value=None)
                    chunk_ms = gr.Number(label="Chunk (ms)", value=20)

                realtime_mode = gr.Checkbox(label="Realtime Simulation (sleep)", value=False)

                with gr.Row():
                    btn_run = gr.Button("▶ RUN", variant="primary", interactive=False)
                    btn_stop = gr.Button("■ STOP", elem_classes=["stop-btn"])

                run_status = gr.Textbox(label="Run Status", value="—", interactive=False, lines=1)

                gr.HTML('<div class="section-header">// System Prompt</div>')
                gr.Textbox(
                    value=SYSTEM_PROMPT[:500] + "...",
                    label="Active System Prompt",
                    lines=5,
                    interactive=False,
                )

            # ── RIGHT: Output ──────────────────────────────────────
            with gr.Column(scale=3):

                with gr.Row():
                    stat_turns     = gr.Textbox(value="0",      label="Turns Done",       interactive=False, lines=1, scale=1)
                    stat_avg_total = gr.Textbox(value="0.000s", label="Avg Total Lat",     interactive=False, lines=1, scale=1)
                    stat_avg_llm   = gr.Textbox(value="0.000s", label="Avg LLM Lat",       interactive=False, lines=1, scale=1)
                    stat_tools     = gr.Textbox(value="0",      label="Total Tool Calls",  interactive=False, lines=1, scale=1)

                with gr.Tabs():
                    # ── Tab 1: Live Monitor ──
                    with gr.Tab("📡 Live Monitor"):
                        log_output = gr.Textbox(
                            label="Event Log",
                            lines=25,
                            interactive=False,
                            elem_classes=["log-terminal"],
                            autoscroll=True,
                        )
                        gr.HTML('<div class="section-header">// Response Audio (Latest)</div>')
                        response_audio = gr.Audio(
                            label="Latest Model Response",
                            interactive=False,
                            autoplay=True,
                            type="filepath",
                        )

                    # ── Tab 2: Metrics ──
                    with gr.Tab("📊 Metrics"):
                        metrics_table = gr.Textbox(
                            label="Latency Breakdown",
                            lines=25,
                            interactive=False,
                            elem_classes=["metrics-table"],
                        )
                        btn_refresh_metrics = gr.Button("↻ Refresh Metrics", variant="secondary")

                    # ── Tab 3: Thinker ──
                    with gr.Tab("🧠 Thinker"):
                        thinker_output = gr.Textbox(
                            label="Model Reasoning Traces",
                            lines=25,
                            interactive=False,
                            elem_classes=["thinker-box"],
                        )
                        btn_refresh_thinker = gr.Button("↻ Refresh Thinker", variant="secondary")

                    # ── Tab 4: Tool Calls ──
                    with gr.Tab("🔧 Tool Calls"):
                        tool_log = gr.JSON(label="Tool Call History", value=[])
                        btn_refresh_tools = gr.Button("↻ Refresh Tool Log", variant="secondary")

                    # ── Tab 5: Raw JSON ──
                    with gr.Tab("📁 Raw Metrics"):
                        raw_json = gr.JSON(label="All Turn Metrics", value=[])
                        btn_refresh_json = gr.Button("↻ Refresh JSON", variant="secondary")

                    # ── Tab 6: Debug ──
                    with gr.Tab("🐛 Debug"):
                        gr.HTML("""
                        <div style="font-family:monospace;font-size:11px;color:#94a3b8;padding:4px 0;margin-bottom:8px;border-bottom:1px solid #2a3347">
                        // Internal debug trace: folder scan, file detection, model input/output
                        </div>
                        """)
                        debug_output = gr.Textbox(
                            label="Debug Trace",
                            lines=25,
                            interactive=False,
                            elem_classes=["debug-box"],
                            autoscroll=True,
                        )
                        btn_refresh_debug = gr.Button("↻ Refresh Debug", variant="secondary")

        # ─────────────────────────────────────────────
        # WIRING
        # ─────────────────────────────────────────────

        btn_init.click(
            fn=init_pipeline,
            inputs=[root_dir, use_mock, model_name, log_dir],
            outputs=[init_status, btn_run],
        )

        root_dir.change(
            fn=get_folder_list,
            inputs=[root_dir],
            outputs=[folder_select],
        )

        btn_run.click(
            fn=run_pipeline,
            inputs=[root_dir, start_idx, end_idx, max_files, chunk_ms, realtime_mode, folder_select],
            outputs=[run_status],
        )

        btn_stop.click(fn=stop_pipeline, outputs=[init_status])

        btn_refresh_metrics.click(fn=manual_refresh_metrics, outputs=[metrics_table])
        btn_refresh_thinker.click(fn=manual_refresh_thinker, outputs=[thinker_output])
        btn_refresh_tools.click(fn=manual_refresh_tools, outputs=[tool_log])
        btn_refresh_json.click(fn=manual_refresh_json, outputs=[raw_json])
        btn_refresh_debug.click(
            fn=lambda: "\n".join(snapshot_state()[2][-200:]),
            outputs=[debug_output],
        )

        # ── refresh outputs list ──
        refresh_outputs = [
            log_output, metrics_table, thinker_output, tool_log, raw_json,
            stat_turns, stat_avg_total, stat_avg_llm, stat_tools,
            response_audio,
            debug_output,
        ]

        demo.load(
            fn=build_refresh_payload,
            outputs=refresh_outputs,
            show_progress="hidden",
        )

        refresh_timer = gr.Timer(2)
        refresh_timer.tick(
            fn=build_refresh_payload,
            outputs=refresh_outputs,
            show_progress="hidden",
        )

    return demo


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(primary_hue="cyan", secondary_hue="slate"),
    )