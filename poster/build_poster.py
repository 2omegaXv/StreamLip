#!/usr/bin/env python3
"""Build the FM-AVSR one-page poster from checked-in assets."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster"
ASSET_DIR = POSTER_DIR / "assets"
TEMPLATE = POSTER_DIR / "Poster Template.pptx"
OUT = POSTER_DIR / "fm_avsr_poster.pptx"


NAVY = RGBColor(18, 35, 61)
TEAL = RGBColor(14, 116, 144)
GREEN = RGBColor(22, 163, 74)
ORANGE = RGBColor(234, 88, 12)
RED = RGBColor(185, 28, 28)
SLATE = RGBColor(51, 65, 85)
MUTED = RGBColor(100, 116, 139)
LIGHT = RGBColor(248, 250, 252)
BORDER = RGBColor(203, 213, 225)
WHITE = RGBColor(255, 255, 255)


def inch(value: float):
    return Inches(value)


def clear_slide(slide):
    sp_tree = slide.shapes._spTree
    for element in list(sp_tree):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"nvGrpSpPr", "grpSpPr"}:
            continue
        sp_tree.remove(element)


def set_text(tf, text, size=24, bold=False, color=SLATE, align=None):
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = "Arial"
    if align is not None:
        p.alignment = align


def add_text(slide, text, x, y, w, h, size=22, bold=False, color=SLATE, align=None):
    box = slide.shapes.add_textbox(inch(x), inch(y), inch(w), inch(h))
    box.text_frame.word_wrap = True
    box.text_frame.margin_left = inch(0.08)
    box.text_frame.margin_right = inch(0.08)
    box.text_frame.margin_top = inch(0.04)
    box.text_frame.margin_bottom = inch(0.04)
    set_text(box.text_frame, text, size, bold, color, align)
    return box


def add_section(slide, title, x, y, w, h):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, inch(x), inch(y), inch(w), inch(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = BORDER
    shape.line.width = Pt(1.2)
    shape.adjustments[0] = 0.06
    add_text(slide, title, x + 0.25, y + 0.18, w - 0.5, 0.45, 21, True, NAVY)
    rule = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(x + 0.25), inch(y + 0.78), inch(w - 0.5), inch(0.035)
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = TEAL
    rule.line.fill.background()
    return shape


def add_bullets(slide, items, x, y, w, h, size=16, color=SLATE):
    box = slide.shapes.add_textbox(inch(x), inch(y), inch(w), inch(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = inch(0.05)
    tf.margin_right = inch(0.05)
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.size = Pt(size)
        p.font.name = "Arial"
        p.font.color.rgb = color
        p.space_after = Pt(5)
        p.line_spacing = 1.05
    return box


def add_metric(slide, label, value, note, x, y, w, color):
    card = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, inch(x), inch(y), inch(w), inch(1.35)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = RGBColor(241, 245, 249)
    card.line.color.rgb = BORDER
    card.adjustments[0] = 0.08
    add_text(slide, value, x + 0.15, y + 0.08, w - 0.3, 0.48, 25, True, color, PP_ALIGN.CENTER)
    add_text(slide, label, x + 0.15, y + 0.58, w - 0.3, 0.32, 11, True, NAVY, PP_ALIGN.CENTER)
    add_text(slide, note, x + 0.15, y + 0.91, w - 0.3, 0.32, 10, False, MUTED, PP_ALIGN.CENTER)


def add_captioned_image(slide, path, caption, x, y, w, h):
    if path.exists():
        slide.shapes.add_picture(str(path), inch(x), inch(y), inch(w), inch(h))
    else:
        missing = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(x), inch(y), inch(w), inch(h)
        )
        missing.fill.solid()
        missing.fill.fore_color.rgb = RGBColor(226, 232, 240)
        missing.line.color.rgb = BORDER
    add_text(slide, caption, x, y + h + 0.08, w, 0.34, 10, True, MUTED, PP_ALIGN.CENTER)


def add_bar_chart(slide, x, y, w, h):
    labels = ["10k", "30k", "+timbre", "+residual", "59k final"]
    values = [0.5073, 0.5308, 0.5633, 0.5726, 0.5818]
    max_v = 0.60
    bar_w = w / len(values) * 0.62
    gap = w / len(values) * 0.38
    for i, (label, value) in enumerate(zip(labels, values)):
        bx = x + i * (bar_w + gap) + gap * 0.5
        bh = h * value / max_v
        by = y + h - bh
        bar = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(bx), inch(by), inch(bar_w), inch(bh)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = GREEN if i == len(values) - 1 else TEAL
        bar.line.fill.background()
        add_text(slide, f"{value:.3f}", bx - 0.05, by - 0.42, bar_w + 0.1, 0.28, 9, True, NAVY, PP_ALIGN.CENTER)
        add_text(slide, label, bx - 0.15, y + h + 0.1, bar_w + 0.3, 0.34, 9, False, MUTED, PP_ALIGN.CENTER)
    axis = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(x), inch(y + h), inch(w), inch(0.025)
    )
    axis.fill.solid()
    axis.fill.fore_color.rgb = BORDER
    axis.line.fill.background()


def add_pipeline(slide, x, y, w, h):
    steps = [
        ("silent video", TEAL),
        ("AVSR visual\n+ weak text", RGBColor(37, 99, 235)),
        ("timbre ref\n3.04 s", ORANGE),
        ("residual\nMimi recon", GREEN),
        ("speech video", NAVY),
    ]
    box_w = w / 5.0 - 0.12
    for i, (label, color) in enumerate(steps):
        bx = x + i * (box_w + 0.15)
        node = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, inch(bx), inch(y), inch(box_w), inch(h)
        )
        node.fill.solid()
        node.fill.fore_color.rgb = color
        node.line.fill.background()
        node.adjustments[0] = 0.12
        add_text(slide, label, bx + 0.05, y + 0.18, box_w - 0.1, h - 0.2, 15, True, WHITE, PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            add_text(slide, ">", bx + box_w - 0.01, y + 0.33, 0.22, 0.32, 16, True, MUTED, PP_ALIGN.CENTER)


def add_condition_table(slide, x, y, w, h):
    rows = [
        ("lip_avsr.npy", "Auto-AVSR lip crop latent"),
        ("avsr_enc_lipavsr.npy", "AVSR visual encoder state"),
        ("avsr_text_lipavsr.txt", "noisy text, weak semantic aid"),
        ("SmolLM2 hidden", "aligned text-token hidden states"),
        ("speaker/timbre", "speaker_emb + Mimi prompt stats"),
        ("target", "Mimi audio latent to reconstruct"),
    ]
    row_h = h / len(rows)
    for i, (name, desc) in enumerate(rows):
        ry = y + i * row_h
        bg = RGBColor(248, 250, 252) if i % 2 == 0 else WHITE
        rect = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(x), inch(ry), inch(w), inch(row_h))
        rect.fill.solid()
        rect.fill.fore_color.rgb = bg
        rect.line.color.rgb = RGBColor(226, 232, 240)
        add_text(slide, name, x + 0.12, ry + 0.08, 2.3, row_h - 0.12, 10, True, NAVY)
        add_text(slide, desc, x + 2.45, ry + 0.08, w - 2.55, row_h - 0.12, 10, False, SLATE)


def build():
    prs = Presentation(str(TEMPLATE))
    slide = prs.slides[0]
    clear_slide(slide)

    sw = prs.slide_width / 914400
    sh = prs.slide_height / 914400

    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = LIGHT

    header = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(0), inch(0), inch(sw), inch(2.28))
    header.fill.solid()
    header.fill.fore_color.rgb = NAVY
    header.line.fill.background()

    footer = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(0), inch(sh - 1.45), inch(sw), inch(1.45))
    footer.fill.solid()
    footer.fill.fore_color.rgb = NAVY
    footer.line.fill.background()

    add_text(slide, "Audio-Centric Visual Speech Reconstruction", 0.75, 0.35, 23.0, 0.7, 31, True, WHITE)
    add_text(slide, "Weak text helps alignment, while lip/audio latents carry the speech signal", 0.78, 1.17, 22.5, 0.42, 16, False, RGBColor(203, 213, 225))
    add_text(slide, "DL-V2A 2026", 25.3, 0.55, 6.8, 0.45, 17, True, WHITE, PP_ALIGN.RIGHT)
    add_text(slide, "Recon model | Auto-AVSR preprocessing | Trump silent-reference demo", 18.5, 1.19, 13.5, 0.38, 12, False, RGBColor(203, 213, 225), PP_ALIGN.RIGHT)

    margin = 0.65
    gap = 0.42
    col_w = (sw - 2 * margin - 2 * gap) / 3
    col1 = margin
    col2 = margin + col_w + gap
    col3 = margin + 2 * (col_w + gap)
    top = 2.65

    add_section(slide, "1. Why audio-centric?", col1, top, col_w, 7.0)
    add_bullets(
        slide,
        [
            "Direct vision -> text -> TTS makes text accuracy the bottleneck.",
            "Our finding: audio quality stays strong when AVSR text is noisy; lip/audio latents matter more.",
            "The model reconstructs Mimi audio latents from visual, text, and timbre conditions.",
        ],
        col1 + 0.35,
        top + 1.0,
        col_w - 0.7,
        3.6,
        17,
    )
    add_metric(slide, "GT text corr", "0.5818", "val1000 final", col1 + 0.35, top + 5.35, 2.9, GREEN)
    add_metric(slide, "AVSR text corr", "0.5783", "drop only 0.0035", col1 + 3.45, top + 5.35, 3.15, TEAL)
    add_metric(slide, "FM corr", "~0.35", "sampling route rejected", col1 + 6.8, top + 5.35, 2.9, ORANGE)

    add_section(slide, "2. Data processing", col1, top + 7.55, col_w, 12.1)
    add_condition_table(slide, col1 + 0.35, top + 8.55, col_w - 0.7, 4.8)
    add_text(
        slide,
        "Reprocessed clips are stored under data/processed/pretrain/*/* with AVSR-compatible lip crops and text. This is a model input decision, not just a cleanup step.",
        col1 + 0.38,
        top + 13.75,
        col_w - 0.75,
        1.15,
        14,
        False,
        SLATE,
    )
    add_text(
        slide,
        "Key point: text is intentionally a weak condition. Even high word-error AVSR text can still preserve intelligible speech when the lip/audio path is reliable.",
        col1 + 0.38,
        top + 15.35,
        col_w - 0.75,
        1.15,
        14,
        True,
        NAVY,
    )
    add_bullets(
        slide,
        [
            "Training/eval data path: raw clip -> face/lip preprocessing -> Auto-AVSR visual/text -> language hidden states -> Mimi audio latent target.",
            "Silent demo path: silent MP4 plus ref audio/video segment; the first 3.04 s reference is encoded as audio prompt and timbre statistics.",
            "The same preprocessing script is used for raw-video inference so evaluation inputs match the new lip_avsr training representation.",
        ],
        col1 + 0.35,
        top + 17.15,
        col_w - 0.7,
        2.15,
        12,
    )

    add_section(slide, "3. Recon architecture", col2, top, col_w, 11.1)
    add_pipeline(slide, col2 + 0.35, top + 1.05, col_w - 0.7, 1.0)
    arch = ASSET_DIR / "system_architecture.png"
    add_captioned_image(slide, arch, "Condition fusion into Mimi latent reconstruction", col2 + 0.35, top + 2.35, col_w - 0.7, 6.2)
    add_text(
        slide,
        "The decoder predicts normalized Mimi latents, which are decoded by the audio codec and muxed back to video. The video stream is preserved; only the speech track is reconstructed.",
        col2 + 0.45,
        top + 9.05,
        col_w - 0.9,
        1.0,
        13,
        False,
        SLATE,
    )

    add_section(slide, "4. Mathematical form", col2, top + 11.55, col_w, 8.4)
    add_text(slide, "C = {visual, text, speaker, audio_prompt, timbre}", col2 + 0.42, top + 12.52, col_w - 0.84, 0.45, 18, True, TEAL)
    add_text(slide, "y_base = f_base(C)", col2 + 0.42, top + 13.25, col_w - 0.84, 0.42, 19, False, SLATE)
    add_text(slide, "delta = f_residual(C)", col2 + 0.42, top + 13.85, col_w - 0.84, 0.42, 19, False, SLATE)
    add_text(slide, "y_hat = y_base + delta", col2 + 0.42, top + 14.47, col_w - 0.84, 0.48, 21, True, GREEN)
    add_text(
        slide,
        "Current inference uses x_tilde = 0 and tau = 1. There is no iterative denoise loop; this is a deterministic one-step residual reconstruction of the target Mimi latent.",
        col2 + 0.42,
        top + 15.35,
        col_w - 0.84,
        1.25,
        14,
        False,
        SLATE,
    )
    add_text(
        slide,
        "Why this matters: it avoids the unstable FM sampling route and directly optimizes the latent representation used by the audio codec.",
        col2 + 0.42,
        top + 17.0,
        col_w - 0.84,
        0.8,
        14,
        True,
        NAVY,
    )
    add_text(
        slide,
        "Interpretation: no Gaussian noise state is sampled during inference; tau is kept at 1 during training and evaluation for this endpoint objective.",
        col2 + 0.42,
        top + 18.25,
        col_w - 0.84,
        0.65,
        12,
        False,
        MUTED,
    )

    add_section(slide, "5. Evaluation trend", col2, top + 20.35, col_w, 10.3)
    add_bar_chart(slide, col2 + 0.65, top + 21.65, col_w - 1.3, 5.0)
    add_text(
        slide,
        "Corr improved from the 10k baseline to the final 59k timbre-conditioned model. The AVSR-text run barely drops, supporting the weak-text condition hypothesis.",
        col2 + 0.45,
        top + 27.5,
        col_w - 0.9,
        0.95,
        14,
        False,
        SLATE,
    )
    add_metric(slide, "10k baseline", "0.507", "small data", col2 + 0.55, top + 29.05, 2.7, TEAL)
    add_metric(slide, "30k + timbre", "0.563", "speaker style", col2 + 3.55, top + 29.05, 3.0, GREEN)
    add_metric(slide, "final 59k", "0.582", "best model", col2 + 6.85, top + 29.05, 2.8, NAVY)

    add_section(slide, "6. Trump silent-reference demo", col3, top, col_w, 17.8)
    add_text(
        slide,
        "Input: 26.57 s silent Trump video. Reference: final 3.02 s audio segment. Output: 23.55 s generated post-prompt video.",
        col3 + 0.35,
        top + 0.95,
        col_w - 0.7,
        0.8,
        14,
        False,
        SLATE,
    )
    img_w = (col_w - 1.0) / 3
    add_captioned_image(slide, ASSET_DIR / "trump_silent_frame.png", "silent input", col3 + 0.35, top + 2.05, img_w, img_w)
    add_captioned_image(slide, ASSET_DIR / "trump_ref_frame.png", "reference", col3 + 0.5 + img_w, top + 2.05, img_w, img_w)
    add_captioned_image(slide, ASSET_DIR / "trump_generated_frame.png", "generated crop", col3 + 0.65 + 2 * img_w, top + 2.05, img_w, img_w)
    add_captioned_image(slide, ASSET_DIR / "trump_ref_wave.png", "reference waveform", col3 + 0.45, top + 5.75, col_w - 0.9, 1.2)
    add_captioned_image(slide, ASSET_DIR / "trump_generated_wave.png", "generated waveform", col3 + 0.45, top + 7.55, col_w - 0.9, 1.2)
    add_text(
        slide,
        "AVSR text example: \"OUR COUNTRY IS WINNING AND IN FACT WE'RE WINNING SO MUCH ...\"",
        col3 + 0.45,
        top + 9.75,
        col_w - 0.9,
        0.85,
        15,
        True,
        NAVY,
    )
    add_text(
        slide,
        "The listening export drops the first 3.04 s because the current timbre prompt can be copied at the beginning.",
        col3 + 0.45,
        top + 11.0,
        col_w - 0.9,
        0.85,
        13,
        False,
        RED,
    )
    add_bullets(
        slide,
        [
            "Recommended ref mode: use an unmasked segment from the same video when available.",
            "Zero-ref mode runs, but timbre is weaker and less speaker-specific.",
            "The generated asset is checked in under data/assets/trump_silent_ref_demo/.",
        ],
        col3 + 0.45,
        top + 12.35,
        col_w - 0.9,
        2.35,
        13,
    )
    add_text(
        slide,
        "Demo phrase: win-win-win / \"WE'RE WINNING SO MUCH\"",
        col3 + 0.45,
        top + 15.55,
        col_w - 0.9,
        0.6,
        14,
        True,
        TEAL,
    )

    add_section(slide, "7. What is new?", col3, top + 18.25, col_w, 7.0)
    add_bullets(
        slide,
        [
            "Residual endpoint reconstruction replaces FM denoise sampling.",
            "Timbre/reference conditioning improves speaker style over plain regression.",
            "Text is demoted to an auxiliary alignment signal instead of the primary content bottleneck.",
            "AVSR-compatible data preprocessing makes raw-video inference match training inputs.",
        ],
        col3 + 0.35,
        top + 19.2,
        col_w - 0.7,
        4.2,
        14,
    )
    add_text(
        slide,
        "The contribution is the full architecture and evidence: data-compatible AVSR latents, weak text conditioning, residual endpoint recon, and a reproducible silent-video demo.",
        col3 + 0.38,
        top + 23.55,
        col_w - 0.76,
        0.75,
        12,
        True,
        NAVY,
    )

    add_section(slide, "8. Current limitation", col3, top + 25.7, col_w, 5.55)
    add_text(
        slide,
        "The audio_prompt is temporal: (38, 512) Mimi frames from the first 3.04 s. It provides strong timbre control, but the model can learn to copy prompt content. Next step: fixed-size timbre embedding or random prompt dropout/windowing during training.",
        col3 + 0.38,
        top + 26.65,
        col_w - 0.76,
        1.55,
        13,
        False,
        SLATE,
    )
    add_bullets(
        slide,
        [
            "Current mitigation: crop first 3.04 s from listening exports.",
            "Better direction: speaker-only timbre embedding independent of utterance time.",
            "Training idea: random reference windows, content dropout, or contrastive speaker loss.",
        ],
        col3 + 0.38,
        top + 28.55,
        col_w - 0.76,
        1.75,
        12,
    )

    add_section(slide, "9. End-to-end artifacts", col1, top + 20.1, col_w, 8.2)
    add_bullets(
        slide,
        [
            "scripts/run_raw_video_avsr_recon_pipeline.py runs preprocessing, AVSR text/latent extraction, recon inference, Mimi decode, and muxing.",
            "scripts/gradio_avsr_gui.py exposes the same flow in a simple GUI.",
            "README documents normal video mode, silent/ref mode, and the current prompt-cropping behavior.",
            "poster/build.sh rebuilds this poster into PPTX, PDF, and PNG preview.",
        ],
        col1 + 0.35,
        top + 21.05,
        col_w - 0.7,
        4.7,
        13,
    )
    add_text(
        slide,
        "Submission-ready example: silent MP4 + 3 s reference -> voiced MP4. The Trump demo is self-contained and reproducible from checked-in assets.",
        col1 + 0.38,
        top + 26.2,
        col_w - 0.76,
        1.0,
        13,
        True,
        NAVY,
    )

    add_section(slide, "10. Main takeaway", col1, top + 28.8, col_w, 8.6)
    add_text(
        slide,
        "We should not treat lip video as a text-recognition problem followed by speech synthesis.",
        col1 + 0.38,
        top + 29.8,
        col_w - 0.76,
        0.75,
        18,
        True,
        TEAL,
    )
    add_bullets(
        slide,
        [
            "Lip motion provides phonetic and timing evidence.",
            "Mimi latents preserve acoustic structure and codec-compatible speech detail.",
            "Text tokens provide alignment and coarse semantics, but do not need to be perfect.",
            "Timbre reference improves voice identity, while exposing a prompt-leakage limitation to fix.",
        ],
        col1 + 0.35,
        top + 31.1,
        col_w - 0.7,
        3.6,
        14,
    )
    add_text(
        slide,
        "Result: a practical raw-video restoration pipeline with stronger audio quality than the FM sampling branch and much lower dependence on exact ASR text.",
        col1 + 0.38,
        top + 35.6,
        col_w - 0.76,
        1.0,
        14,
        True,
        NAVY,
    )

    add_section(slide, "11. Why not cascade?", col2, top + 31.15, col_w, 6.25)
    add_bullets(
        slide,
        [
            "Cascade: lip -> text -> TTS must solve exact word recovery before generating audio.",
            "Our route: lip/audio latent reconstruction can tolerate noisy text because visual and audio conditions dominate.",
            "Empirical check: replacing GT text with AVSR text changes corr from 0.5818 to 0.5783.",
        ],
        col2 + 0.35,
        top + 32.05,
        col_w - 0.7,
        3.35,
        14,
    )
    add_text(
        slide,
        "This is the central architectural argument: use text as support, not as the only bridge between video and sound.",
        col2 + 0.38,
        top + 35.75,
        col_w - 0.76,
        0.75,
        14,
        True,
        NAVY,
    )

    add_section(slide, "12. Files to show", col3, top + 31.75, col_w, 5.65)
    add_bullets(
        slide,
        [
            "poster/fm_avsr_poster.pptx",
            "poster/fm_avsr_poster.pdf",
            "poster/fm_avsr_poster_preview.png",
            "data/assets/trump_silent_ref_demo/*.mp4",
        ],
        col3 + 0.35,
        top + 32.65,
        col_w - 0.7,
        2.6,
        14,
    )
    add_text(
        slide,
        "All paths are relative to the fm-avsr-cleanup worktree.",
        col3 + 0.38,
        top + 35.65,
        col_w - 0.76,
        0.6,
        12,
        False,
        MUTED,
    )

    add_text(slide, "Deep Learning 2026 Spring", 0.75, sh - 1.03, 8.0, 0.42, 14, True, WHITE)
    add_text(slide, "DL-V2A | FM-AVSR cleanup branch | Generated from poster/build_poster.py", 9.0, sh - 1.03, 23.2, 0.42, 12, False, RGBColor(203, 213, 225), PP_ALIGN.RIGHT)

    prs.save(str(OUT))
    print(OUT)


if __name__ == "__main__":
    build()
