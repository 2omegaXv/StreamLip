#!/usr/bin/env python3
"""Gradio GUI for the Raw Video AVSR Reconstruction Pipeline.

    /mnt/pfs/group-jt/zihan.guo/droid/DL-V2A/.venv/bin/python \\
        scripts/gradio_avsr_gui.py [--port 7860]
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


# Unset SOCKS proxy before gradio / httpx are imported.
# The ALL_PROXY env var is set system-wide but points to a local tunnel that
# blocks httpx's localhost reachability check inside gradio.launch().
import os as _os
for _k in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY",
           "http_proxy", "https_proxy"):
    _os.environ.pop(_k, None)
_os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# Disable Google Fonts CDN to avoid browser hang on restricted networks
_os.environ["GRADIO_FONTS_CSS"] = ""
_os.environ.setdefault("GRADIO_CDN_FONTS", "0")

import gradio as gr
import gradio.networking as _gnet

# Patch url_ok so gradio.launch() doesn't hang trying to reach localhost
# through a dead SOCKS proxy.
_gnet.url_ok = lambda url, **kw: True   # noqa: E731

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path("/mnt/pfs/group-jt/zihan.guo/droid/DL-V2A")
VENV_PYTHON = MAIN_ROOT / ".venv/bin/python"
PIPELINE_SCRIPT = REPO_ROOT / "scripts/run_raw_video_avsr_recon_pipeline.py"

DEFAULT_CONFIG = (
    "configs/fm_avsr_lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_"
    "residual_samplecorr02_from1000_recon_textjson_wordts.yaml"
)
DEFAULT_CKPT = (
    "runs/fm_avsr/lipavsr_59144_timbre3s_audioprompt38_pool_promptstats005_"
    "residual_samplecorr02_from1000_recon_textjson_wordts_v1/step_001500.pt"
)

# ---------------------------------------------------------------------------
# Steps: (keyword_in_stdout, label)
# ---------------------------------------------------------------------------
STEPS = [
    ("ffmpeg",                "标准化视频"),
    ("run_preprocess_worker", "人脸 / 音频预处理"),
    ("reprocess_worker_avsr", "AVSR 嘴唇裁剪"),
    ("extract_latent",        "提取 Mimi 潜变量"),
    ("extract_avsr_enc",      "AVSR 编码器 + 文本识别"),
    ("extract_smollm2",       "SmolLM2 特征提取"),
    ("extract_speaker",       "说话人 / 音色条件"),
    ("eval_fm_avsr",          "FM-AVSR 重建推理"),
    ("mux",                   "合成输出视频"),
]
N = len(STEPS)

TICK = 0.5   # seconds between timer refreshes


def _clean_env() -> dict:
    env = os.environ.copy()
    for k in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "http_proxy", "https_proxy"):
        env.pop(k, None)
    return env


def _infer_step(line: str, current: int) -> int:
    for i in range(current, N):
        if STEPS[i][0] in line:
            return i
    return current


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _progress_html(step: int, elapsed: float, finished: bool, error: bool = False) -> str:
    time_str = _fmt_time(elapsed)
    if error:
        bar_color = "#ef4444"
        label = f"❌ 失败（{STEPS[min(step, N-1)][1]}）"
    elif finished:
        bar_color = "#16a34a"
        label = "✅ 完成"
    else:
        bar_color = "#3b82f6"
        label = f"{STEPS[min(step, N-1)][1]} …"

    running = not finished and not error
    anim_style = (
        "<style>"
        "@keyframes stripe{"
        "from{background-position:0 0}to{background-position:40px 0}}"
        ".avsr-bar{"
        "animation:stripe 1s linear infinite;"
        "background-image:repeating-linear-gradient("
        "45deg,transparent,transparent 10px,"
        "rgba(255,255,255,.18) 10px,rgba(255,255,255,.18) 20px)}"
        "</style>"
    ) if running else ""

    bar_class = "avsr-bar" if running else ""

    return (
        f"{anim_style}"
        f'<div style="font-family:sans-serif;padding:4px 0">'
        f'  <div style="display:flex;justify-content:space-between;'
        f'              align-items:baseline;margin-bottom:6px">'
        f'    <span style="font-size:0.95em;color:#374151">{label}</span>'
        f'    <span style="font-size:0.92em;font-family:monospace;'
        f'                 color:#6b7280">{time_str}</span>'
        f'  </div>'
        f'  <div style="background:#e5e7eb;border-radius:999px;height:10px;overflow:hidden">'
        f'    <div class="{bar_class}"'
        f'         style="width:100%;height:100%;background:{bar_color};'
        f'                border-radius:999px"></div>'
        f'  </div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Pipeline generator
# yields (btn_update, progress_html, result_video)
# ---------------------------------------------------------------------------
def run_pipeline(
    input_video,
    ref_audio,
    silent_input,
    exp_name,
    fa_device,
    size,
    force,
    mimi_path,
    smollm2_path,
    auto_avsr_ckpt,
    resnet50_weights,
    norm_stats,
    config,
    ckpt,
):
    btn_running  = gr.update(value="⏳ 运行中…", interactive=False, variant="secondary")
    btn_idle     = gr.update(value="▶ 运行",      interactive=True,  variant="primary")

    def _err(msg, elapsed=0.0):
        html = (
            f'<div style="font-family:sans-serif;padding:4px 0;color:#ef4444">'
            f'{msg}</div>'
        )
        yield btn_idle, html, None

    if not input_video:
        yield from _err("❌ 请先上传视频文件")
        return
    if not exp_name.strip():
        yield from _err("❌ 请填写实验名称")
        return

    exp = exp_name.strip().replace("/", "_")

    cmd = [
        str(VENV_PYTHON), str(PIPELINE_SCRIPT),
        "--input",            str(input_video),
        "--exp",              exp,
        "--fa_device",        fa_device,
        "--size",             str(int(size)),
        "--mimi_path",        mimi_path,
        "--smollm2_path",     smollm2_path,
        "--auto_avsr_ckpt",   auto_avsr_ckpt,
        "--resnet50_weights", resnet50_weights,
        "--norm_stats",       norm_stats,
        "--config",           config,
        "--ckpt",             ckpt,
    ]
    if silent_input:
        cmd.append("--silent_input")
    if ref_audio:
        cmd.extend(["--ref_audio", str(ref_audio)])
    if force:
        cmd.append("--force")

    # --- start ---
    step = 0
    t0   = time.time()
    yield btn_running, _progress_html(step, 0.0, False), None

    # read stdout in background thread → push new step indices to a queue
    step_q = queue.Queue()

    def _reader(proc):
        cur = 0
        for line in proc.stdout:
            new = _infer_step(line, cur)
            if new != cur:
                cur = new
                step_q.put(cur)
        step_q.put(None)   # sentinel: process done

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(REPO_ROOT),
        env=_clean_env(),
        bufsize=1,
    )
    t = threading.Thread(target=_reader, args=(proc,), daemon=True)
    t.start()

    # --- tick loop: yield every TICK seconds for live timer ---
    done = False
    while not done:
        time.sleep(TICK)
        # drain all pending step updates
        while True:
            try:
                item = step_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                done = True
                break
            step = item
        yield btn_running, _progress_html(step, time.time() - t0, False), None

    t.join()
    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        yield btn_idle, _progress_html(step, elapsed, False, error=True), None
        return

    pred_name = f"{exp}_pred_full.mp4" if silent_input else f"{exp}_pred_prompt3s_post3s.mp4"
    pred_mp4 = REPO_ROOT / "eval_out" / exp / pred_name
    video_out = str(pred_mp4) if pred_mp4.exists() else None
    yield btn_idle, _progress_html(N - 1, elapsed, finished=True), video_out


_INIT_JS = """
() => {
    console.log('[AVSR GUI] page loaded, gradio_config:', window.gradio_config?.version);

    // intercept WebSocket to log connection events
    const OrigWS = window.WebSocket;
    window.WebSocket = function(url, ...rest) {
        const ws = new OrigWS(url, ...rest);
        console.log('[AVSR GUI] WebSocket open attempt:', url);
        ws.addEventListener('open',    () => console.log('[AVSR GUI] WS opened:', url));
        ws.addEventListener('error', e => console.error('[AVSR GUI] WS error:', url, e));
        ws.addEventListener('close', e => console.warn('[AVSR GUI] WS closed:', url, 'code:', e.code, e.reason));
        ws.addEventListener('message', e => {
            try {
                const d = JSON.parse(e.data);
                if (d.msg) console.log('[AVSR GUI] WS msg:', d.msg, d);
            } catch(_) {}
        });
        return ws;
    };
    window.WebSocket.prototype = OrigWS.prototype;
    window.WebSocket.CONNECTING = OrigWS.CONNECTING;
    window.WebSocket.OPEN       = OrigWS.OPEN;
    window.WebSocket.CLOSING    = OrigWS.CLOSING;
    window.WebSocket.CLOSED     = OrigWS.CLOSED;

    // log fetch errors
    const origFetch = window.fetch;
    window.fetch = function(input, ...rest) {
        const url = typeof input === 'string' ? input : input?.url;
        return origFetch(input, ...rest).then(r => {
            if (!r.ok) console.error('[AVSR GUI] fetch error:', url, r.status, r.statusText);
            return r;
        }).catch(err => {
            console.error('[AVSR GUI] fetch threw:', url, err);
            throw err;
        });
    };

    console.log('[AVSR GUI] instrumentation installed');
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AVSR Recon", js=_INIT_JS) as demo:

        gr.Markdown(
            "## 🎙 Raw Video → AVSR 音频重建\n"
            "上传视频，可选 reference 音频作为音色 prompt，输出为生成音频合成回视频。"
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                input_video = gr.Video(label="输入视频 (.mov / .mp4)", sources=["upload"])
                ref_audio   = gr.Audio(label="参考音频（可选）", sources=["upload"], type="filepath")
                silent_input = gr.Checkbox(label="输入视频无声音 / 输出保留整段长度", value=False)
                exp_name    = gr.Textbox(label="实验名称", placeholder="如 trump_test")
                force       = gr.Checkbox(label="强制重跑（清除旧结果）", value=True)

                with gr.Accordion("⚙️ 高级参数", open=False):
                    fa_device        = gr.Textbox(label="fa_device",     value="cuda:0")
                    size             = gr.Number( label="缩放尺寸 (px)", value=224, precision=0)
                    mimi_path        = gr.Textbox(label="Mimi",          value=str(MAIN_ROOT / "pretrained/mimi"))
                    smollm2_path     = gr.Textbox(label="SmolLM2",       value=str(MAIN_ROOT / "pretrained/smollm2-360m"))
                    auto_avsr_ckpt   = gr.Textbox(label="Auto-AVSR ckpt",value=str(MAIN_ROOT / "pretrained/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth"))
                    resnet50_weights = gr.Textbox(label="ResNet50 权重", value=str(MAIN_ROOT / "pretrained/resnet50-11ad3fa6.pth"))
                    norm_stats       = gr.Textbox(label="Norm stats",    value=str(MAIN_ROOT / "data/processed/latent_norm_stats.npz"))
                    config_path      = gr.Textbox(label="Config yaml",   value=DEFAULT_CONFIG)
                    ckpt_path        = gr.Textbox(label="Checkpoint",    value=DEFAULT_CKPT)

                run_btn  = gr.Button("▶ 运行", variant="primary", size="lg")
                progress = gr.HTML(value=_progress_html(0, 0.0, False))

            with gr.Column(scale=2):
                result_video = gr.Video(
                    label="生成结果（生成音频 + 原始视频）",
                    interactive=False,
                )

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                input_video, ref_audio, silent_input, exp_name, fa_device, size, force,
                mimi_path, smollm2_path, auto_avsr_ckpt, resnet50_weights,
                norm_stats, config_path, ckpt_path,
            ],
            outputs=[run_btn, progress, result_video],
            show_progress="hidden",
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app()
    demo.queue()
    print(f"\n🚀  http://{args.host}:{args.port}\n", flush=True)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(REPO_ROOT), str(MAIN_ROOT)],
    )


if __name__ == "__main__":
    main()
