"""
app.py
──────
Gradio UI cho Audio Conversation Test System.
Theme: Industrial / Utilitarian dark — monospace, dense info, terminal feel.
"""

import gradio as gr
import threading
import queue
import time
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import AudioTestPipeline, PipelineEvent, get_gen_folders, get_user_files
from core.model_client import SYSTEM_PROMPT

logger = logging.getLogger("gradio_app")

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────

pipeline_instance: AudioTestPipeline = None
event_queue: queue.Queue = queue.Queue()
run_thread: threading.Thread = None
current_metrics: list[dict] = []
log_lines: list[str] = []

MAX_LOG_LINES = 300

def add_log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    icon = {"DEBUG": "·", "INFO": "►", "WARN": "⚠", "ERROR": "✖", "SUCCESS": "✔"}.get(level, "·")
    line = f"[{ts}] {icon} {msg}"
    log_lines.append(line)
    if len(log_lines) > MAX_LOG_LINES:
        log_lines.pop(0)


# ─────────────────────────────────────────────
# EVENT HANDLER
# ─────────────────────────────────────────────

def on_pipeline_event(event: PipelineEvent):
    event_queue.put(event)
    
    etype = event.type
    p = event.payload
    tid = event.turn_id or event.conversation_id

    if etype == "conversation_start":
        add_log(f"📁 START CONVERSATION: {event.conversation_id} ({p.get('total_turns',0)} turns)", "INFO")
    elif etype == "turn_start":
        add_log(f"┌─ TURN {p.get('turn_num','')+1}: {p.get('audio_file','')}", "INFO")
    elif etype == "asr_done":
        add_log(f"│  ASR [{p.get('latency_s',0):.3f}s]: \"{p.get('transcript','')}\"", "INFO")
    elif etype == "llm_done":
        thinker = p.get("thinker", "")
        lat = p.get("latency_s", 0)
        if thinker:
            add_log(f"│  🧠 THINKER [{lat:.3f}s]: {thinker[:100]}...", "DEBUG")
        else:
            add_log(f"│  LLM [{lat:.3f}s]: {p.get('clean','')[:100]}", "INFO")
    elif etype == "tool_call":
        calls = p.get("calls", [])
        for c in calls:
            add_log(f"│  🔧 TOOL CALL: {c['name']}({json.dumps(c.get('arguments',{}), ensure_ascii=False)[:80]})", "WARN")
    elif etype == "tool_result":
        for r in p.get("results", []):
            result_preview = str(r.get("result", r.get("error", "")))[:80]
            add_log(f"│  ✔ TOOL RESULT [{p.get('latency_s',0):.3f}s]: {result_preview}", "SUCCESS")
    elif etype == "tts_done":
        add_log(f"│  🔊 TTS [{p.get('latency_s',0):.3f}s]: {p.get('response_text','')[:80]}", "INFO")
    elif etype == "turn_end":
        m = p
        add_log(
            f"└─ DONE | total={m.get('total_latency_s',0):.3f}s "
            f"asr={m.get('audio_encode_latency_s',0):.3f}s "
            f"llm={m.get('inference_latency_s',0):.3f}s "
            f"tool={m.get('tool_latency_s',0):.3f}s "
            f"tts={m.get('tts_latency_s',0):.3f}s",
            "SUCCESS"
        )
        current_metrics.append(p)
    elif etype == "error":
        add_log(f"✖ ERROR [{tid}]: {p.get('error','')}", "ERROR")
    elif etype == "conversation_end":
        add_log(f"✔ CONVERSATION DONE: {event.conversation_id} ({p.get('total_turns',0)} turns)\n", "SUCCESS")


# ─────────────────────────────────────────────
# PIPELINE CONTROL
# ─────────────────────────────────────────────

def init_pipeline(root_dir: str, use_mock: bool, model_name: str, log_dir: str):
    global pipeline_instance, current_metrics, log_lines
    current_metrics = []
    log_lines = []
    
    if not root_dir or not os.path.isdir(root_dir):
        return "❌ Root directory không tồn tại!", gr.update(interactive=False)

    try:
        pipeline_instance = AudioTestPipeline(
            root_dir=root_dir,
            log_dir=log_dir or "./logs",
            use_mock=use_mock,
            model_name=model_name or "thaomike/qwen-finetune-research",
            on_event=on_pipeline_event,
        )
        folders = get_gen_folders(root_dir)
        add_log(f"Pipeline initialized | {len(folders)} folders found | mock={use_mock}", "SUCCESS")
        return f"✔ Pipeline ready | {len(folders)} conversations found", gr.update(interactive=True)
    except Exception as e:
        add_log(f"Init failed: {e}", "ERROR")
        return f"❌ Init failed: {e}", gr.update(interactive=False)


def run_pipeline(
    root_dir, start_idx, end_idx, max_files,
    chunk_ms, realtime_mode, selected_folder
):
    global run_thread, pipeline_instance, current_metrics

    if pipeline_instance is None:
        add_log("Pipeline not initialized!", "ERROR")
        return

    pipeline_instance._stop_flag.clear()
    current_metrics = []

    def _run():
        try:
            if selected_folder and selected_folder != "ALL":
                add_log(f"Running single conversation: {selected_folder}", "INFO")
                pipeline_instance.run_conversation(
                    selected_folder,
                    max_files=int(max_files) if max_files else None
                )
            else:
                add_log(f"Running all conversations [{start_idx}→{end_idx}]", "INFO")
                pipeline_instance.run_all(
                    start=int(start_idx or 0),
                    end=int(end_idx) if end_idx else None,
                    max_files=int(max_files) if max_files else None,
                )
        except Exception as e:
            add_log(f"Run error: {e}", "ERROR")

    run_thread = threading.Thread(target=_run, daemon=True)
    run_thread.start()


def stop_pipeline():
    if pipeline_instance:
        pipeline_instance.stop()
        add_log("⚡ Stop requested", "WARN")
    return "⚡ Stopping..."


def get_folder_list(root_dir: str):
    if not root_dir or not os.path.isdir(root_dir):
        return gr.update(choices=["ALL"])
    folders = get_gen_folders(root_dir)
    return gr.update(choices=["ALL"] + folders, value="ALL")


def format_metrics_table(metrics: list[dict]) -> str:
    if not metrics:
        return "No data yet."
    
    lines = ["TURN_ID                   | AUDIO  | ASR    | LLM    | TOOL   | TTS    | TOTAL  | TOOLS | ERR"]
    lines.append("─" * 105)
    for m in metrics[-50:]:  # last 50
        tid = m.get("turn_id", "")[:25].ljust(25)
        aud = f"{m.get('audio_duration_s',0):.2f}s".ljust(6)
        asr = f"{m.get('audio_encode_latency_s',0):.3f}s".ljust(6)
        llm = f"{m.get('inference_latency_s',0):.3f}s".ljust(6)
        tool = f"{m.get('tool_latency_s',0):.3f}s".ljust(6)
        tts = f"{m.get('tts_latency_s',0):.3f}s".ljust(6)
        total = f"{m.get('total_latency_s',0):.3f}s".ljust(6)
        tools = ", ".join(t["name"] for t in m.get("tool_calls", []))[:20].ljust(20)
        err = "✖" if m.get("error") else "·"
        lines.append(f"{tid} | {aud} | {asr} | {llm} | {tool} | {tts} | {total} | {tools} | {err}")
    
    if len(metrics) > 0:
        lines.append("─" * 105)
        avgs = {
            "asr": sum(m.get("audio_encode_latency_s",0) for m in metrics)/len(metrics),
            "llm": sum(m.get("inference_latency_s",0) for m in metrics)/len(metrics),
            "tool": sum(m.get("tool_latency_s",0) for m in metrics)/len(metrics),
            "tts": sum(m.get("tts_latency_s",0) for m in metrics)/len(metrics),
            "total": sum(m.get("total_latency_s",0) for m in metrics)/len(metrics),
        }
        lines.append(
            f"{'AVG (' + str(len(metrics)) + ' turns)':25s} | {'':6s} | "
            f"{avgs['asr']:.3f}s | {avgs['llm']:.3f}s | {avgs['tool']:.3f}s | "
            f"{avgs['tts']:.3f}s | {avgs['total']:.3f}s"
        )
    
    return "\n".join(lines)


def format_thinker_log(metrics: list[dict]) -> str:
    if not metrics:
        return "No thinker data."
    output = []
    for m in metrics:
        t = m.get("thinker_text", "")
        if t:
            output.append(f"── {m.get('turn_id','')} ──")
            output.append(t)
            output.append("")
    return "\n".join(output) if output else "No thinker blocks found in processed turns."


def get_latest_audio():
    """Trả về audio response mới nhất."""
    if current_metrics:
        for m in reversed(current_metrics):
            path = m.get("response_audio_path")
            if path and os.path.exists(path):
                return path
    return None


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
  --bg-0: #0a0c0f;
  --bg-1: #10131a;
  --bg-2: #161b26;
  --bg-3: #1e2535;
  --border: #2a3347;
  --accent-cyan: #00d4ff;
  --accent-green: #00ff88;
  --accent-amber: #ffb800;
  --accent-red: #ff4444;
  --accent-purple: #a855f7;
  --text-primary: #e2e8f0;
  --text-secondary: #94a3b8;
  --text-dim: #475569;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'IBM Plex Sans', sans-serif;
}

/* Global Reset */
body, .gradio-container {
  background: var(--bg-0) !important;
  font-family: var(--sans) !important;
  color: var(--text-primary) !important;
}

.gradio-container {
  max-width: 1600px !important;
  margin: 0 auto !important;
}

/* Header */
.app-header {
  background: var(--bg-1);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 0;
}

.app-title {
  font-family: var(--mono);
  font-size: 18px;
  font-weight: 700;
  color: var(--accent-cyan);
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.app-subtitle {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
  letter-spacing: 0.1em;
}

.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--accent-green);
  box-shadow: 0 0 6px var(--accent-green);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

/* Panels */
.panel {
  background: var(--bg-1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}

.panel-label {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  font-weight: 700 !important;
  color: var(--text-dim) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.15em !important;
  padding: 8px 12px 4px !important;
  border-bottom: 1px solid var(--border) !important;
}

/* Inputs */
input[type="text"], input[type="number"], select, textarea {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  font-family: var(--mono) !important;
  font-size: 13px !important;
  border-radius: 3px !important;
}

input:focus, textarea:focus {
  border-color: var(--accent-cyan) !important;
  outline: none !important;
  box-shadow: 0 0 0 1px var(--accent-cyan) !important;
}

/* Buttons */
button.primary {
  background: transparent !important;
  border: 1px solid var(--accent-cyan) !important;
  color: var(--accent-cyan) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border-radius: 3px !important;
  transition: all 0.15s !important;
}

button.primary:hover {
  background: var(--accent-cyan) !important;
  color: var(--bg-0) !important;
  box-shadow: 0 0 12px rgba(0, 212, 255, 0.3) !important;
}

button.secondary {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-secondary) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  letter-spacing: 0.05em !important;
  border-radius: 3px !important;
}

button.stop-btn {
  border-color: var(--accent-red) !important;
  color: var(--accent-red) !important;
}

button.stop-btn:hover {
  background: var(--accent-red) !important;
  color: white !important;
}

/* Log Terminal */
.log-terminal textarea {
  background: #050709 !important;
  color: #a0ffb0 !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  line-height: 1.6 !important;
  border: 1px solid #1a2a1a !important;
  border-radius: 3px !important;
}

/* Metrics Table */
.metrics-table textarea {
  background: var(--bg-1) !important;
  color: var(--text-secondary) !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  line-height: 1.5 !important;
  border-color: var(--border) !important;
}

/* Thinker box */
.thinker-box textarea {
  background: #0f0a1a !important;
  color: #c084fc !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
  border-color: #3b1f6e !important;
}

/* Labels */
label, .label-wrap span {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--text-dim) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}

/* Stat boxes */
.stat-box {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 8px 12px;
  text-align: center;
}

.stat-value {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 700;
  color: var(--accent-cyan);
  display: block;
}

.stat-label {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

/* Section dividers */
.section-header {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.2em;
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
}

/* Tabs */
.tab-nav button {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}

/* Checkbox */
input[type="checkbox"] {
  accent-color: var(--accent-cyan) !important;
}

/* Audio player */
audio {
  width: 100%;
  filter: invert(0.85) hue-rotate(180deg);
}
"""

def build_ui():
    with gr.Blocks(
        title="VoiceTest System",
    ) as demo:
        
        # ── Header ──
        gr.HTML("""
        <div class="app-header">
          <div class="status-dot"></div>
          <div>
            <div class="app-title">◈ VoiceTest System v1.0</div>
            <div class="app-subtitle">Qwen2.5-7B-Omni · Insurance Claim Conversation Tester · Debug Mode</div>
          </div>
        </div>
        """)

        with gr.Row():
            # ── LEFT PANEL: Config ──
            with gr.Column(scale=1, min_width=320):
                
                gr.HTML('<div class="section-header">// Pipeline Config</div>')
                
                root_dir = gr.Textbox(
                    label="Root Directory",
                    placeholder="/storage/huytg/dataset_audio_vieneu_clean_f5tts_v2",
                    value="./sample_data",
                )
                log_dir = gr.Textbox(
                    label="Log Directory",
                    value="./logs",
                )
                model_name = gr.Textbox(
                    label="Model Name (HuggingFace)",
                    value="thaomike/qwen-finetune-research",
                )
                use_mock = gr.Checkbox(
                    label="Use Mock Model (no GPU)",
                    value=True,
                )
                
                btn_init = gr.Button("⚡ INITIALIZE PIPELINE", variant="primary")
                init_status = gr.Textbox(
                    label="Status",
                    value="Not initialized",
                    interactive=False,
                    lines=1,
                )

                gr.HTML('<div class="section-header">// Run Config</div>')
                
                folder_select = gr.Dropdown(
                    label="Conversation (folder)",
                    choices=["ALL"],
                    value="ALL",
                    interactive=True,
                )
                
                with gr.Row():
                    start_idx = gr.Number(label="Start Index", value=0, minimum=0)
                    end_idx = gr.Number(label="End Index", value=None, minimum=0)
                
                with gr.Row():
                    max_files = gr.Number(label="Max Turns/Conv", value=None, minimum=1)
                    chunk_ms = gr.Number(label="Chunk Size (ms)", value=20, minimum=10)
                
                realtime_mode = gr.Checkbox(label="Realtime Simulation (sleep)", value=False)
                
                with gr.Row():
                    btn_run = gr.Button("▶ RUN", variant="primary", interactive=False)
                    btn_stop = gr.Button("■ STOP", elem_classes=["stop-btn"])
                
                gr.HTML('<div class="section-header">// System Prompt Preview</div>')
                gr.Textbox(
                    value=SYSTEM_PROMPT[:500] + "...",
                    label="Active System Prompt",
                    lines=5,
                    interactive=False,
                )

            # ── RIGHT PANEL: Output ──
            with gr.Column(scale=3):
                
                with gr.Tabs():
                    
                    # Tab 1: Live Monitor
                    with gr.Tab("📡 Live Monitor"):
                        with gr.Row():
                            stat_turns = gr.Textbox(value="0", label="Turns Processed", interactive=False)
                            stat_avg_total = gr.Textbox(value="0.000s", label="Avg Total Latency", interactive=False)
                            stat_avg_llm = gr.Textbox(value="0.000s", label="Avg LLM Latency", interactive=False)
                            stat_tools = gr.Textbox(value="0", label="Tool Calls", interactive=False)
                        
                        log_output = gr.Textbox(
                            label="Event Log",
                            lines=25,
                            max_lines=30,
                            interactive=False,
                            elem_classes=["log-terminal"],
                            buttons=["copy"],
                        )
                        
                        gr.HTML('<div class="section-header">// Response Audio (Latest)</div>')
                        response_audio = gr.Audio(
                            label="Latest Model Response",
                            interactive=False,
                            autoplay=True,
                        )
                    
                    # Tab 2: Metrics Table
                    with gr.Tab("📊 Metrics"):
                        metrics_table = gr.Textbox(
                            label="Latency Breakdown Table",
                            lines=25,
                            interactive=False,
                            elem_classes=["metrics-table"],
                            buttons=["copy"],
                        )
                        btn_refresh_metrics = gr.Button("↻ Refresh Metrics", variant="secondary")
                    
                    # Tab 3: Thinker Log
                    with gr.Tab("🧠 Thinker / Reasoning"):
                        thinker_output = gr.Textbox(
                            label="Model Reasoning Traces",
                            lines=25,
                            interactive=False,
                            elem_classes=["thinker-box"],
                            buttons=["copy"],
                        )
                        btn_refresh_thinker = gr.Button("↻ Refresh Thinker", variant="secondary")
                    
                    # Tab 4: Tool Calls
                    with gr.Tab("🔧 Tool Calls"):
                        tool_log = gr.JSON(
                            label="Tool Call History",
                            value=[],
                        )
                        btn_refresh_tools = gr.Button("↻ Refresh Tool Log", variant="secondary")
                    
                    # Tab 5: Raw JSON
                    with gr.Tab("📁 Raw Metrics JSON"):
                        raw_json = gr.JSON(label="All Turn Metrics", value=[])
                        btn_refresh_json = gr.Button("↻ Refresh JSON", variant="secondary")

        # ─────────────────────────────────────────────
        # EVENT BINDINGS
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
            outputs=[],
        )

        btn_stop.click(fn=stop_pipeline, inputs=[], outputs=[init_status])

        # Polling refresh
        def refresh_all():
            logs = "\n".join(log_lines[-100:])
            metrics_text = format_metrics_table(current_metrics)
            thinker_text = format_thinker_log(current_metrics)
            
            n = len(current_metrics)
            avg_total = f"{sum(m.get('total_latency_s',0) for m in current_metrics)/n:.3f}s" if n else "0.000s"
            avg_llm = f"{sum(m.get('inference_latency_s',0) for m in current_metrics)/n:.3f}s" if n else "0.000s"
            tools_count = sum(len(m.get('tool_calls', [])) for m in current_metrics)
            
            tool_history = []
            for m in current_metrics:
                for tc in m.get("tool_calls", []):
                    tool_history.append({
                        "turn": m.get("turn_id"),
                        "tool": tc.get("name"),
                        "args": tc.get("args", {})
                    })
            
            latest_audio = get_latest_audio()
            
            return (
                logs,
                metrics_text,
                thinker_text,
                tool_history,
                current_metrics[-50:] if current_metrics else [],
                str(n),
                avg_total,
                avg_llm,
                str(tools_count),
                latest_audio,
            )

        def manual_refresh_metrics():
            return format_metrics_table(current_metrics)

        def manual_refresh_thinker():
            return format_thinker_log(current_metrics)

        def manual_refresh_tools():
            tool_history = []
            for m in current_metrics:
                for tc in m.get("tool_calls", []):
                    tool_history.append({
                        "turn": m.get("turn_id"),
                        "tool": tc.get("name"),
                        "args": tc.get("args", {})
                    })
            return tool_history

        def manual_refresh_json():
            return current_metrics[-50:] if current_metrics else []

        btn_refresh_metrics.click(fn=manual_refresh_metrics, outputs=[metrics_table])
        btn_refresh_thinker.click(fn=manual_refresh_thinker, outputs=[thinker_output])
        btn_refresh_tools.click(fn=manual_refresh_tools, outputs=[tool_log])
        btn_refresh_json.click(fn=manual_refresh_json, outputs=[raw_json])

        # Initial refresh + auto-refresh every 2 seconds
        demo.load(
            fn=refresh_all,
            outputs=[
                log_output, metrics_table, thinker_output,
                tool_log, raw_json,
                stat_turns, stat_avg_total, stat_avg_llm, stat_tools,
                response_audio,
            ],
        )
        refresh_timer = gr.Timer(2)
        refresh_timer.tick(
            fn=refresh_all,
            outputs=[
                log_output, metrics_table, thinker_output,
                tool_log, raw_json,
                stat_turns, stat_avg_total, stat_avg_llm, stat_tools,
                response_audio,
            ],
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
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="slate",
        ),
    )
