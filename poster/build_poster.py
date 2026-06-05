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


def clear_body_only(slide, slide_height_inches: float):
    """Keep template header/footer artwork; remove editable placeholder body."""
    sp_tree = slide.shapes._spTree
    keep = []
    for shape in slide.shapes:
        y = shape.top / 914400
        bottom = (shape.top + shape.height) / 914400
        text = shape.text.strip() if hasattr(shape, "text") else ""
        is_template_header = y < 2.35
        is_template_footer = bottom > slide_height_inches - 2.05
        is_course_footer = "Deep Learning 2026 Spring" in text
        keep.append(is_template_header or is_template_footer or is_course_footer)

    for element, should_keep in zip(list(sp_tree)[2:], keep):
        if should_keep:
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


def add_panel(slide, title, x, y, w, h, accent=TEAL, fill=WHITE):
    panel = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, inch(x), inch(y), inch(w), inch(h)
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    panel.line.color.rgb = BORDER
    panel.line.width = Pt(1.0)
    panel.adjustments[0] = 0.035
    add_text(slide, title, x + 0.22, y + 0.15, w - 0.44, 0.34, 12, True, accent)
    return panel


def add_tag(slide, text, x, y, w, color=TEAL):
    tag = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, inch(x), inch(y), inch(w), inch(0.42)
    )
    tag.fill.solid()
    tag.fill.fore_color.rgb = color
    tag.line.fill.background()
    tag.adjustments[0] = 0.14
    add_text(slide, text, x + 0.08, y + 0.08, w - 0.16, 0.22, 9, True, WHITE, PP_ALIGN.CENTER)


def add_image(slide, path, x, y, w, h):
    if path.exists():
        return slide.shapes.add_picture(str(path), inch(x), inch(y), inch(w), inch(h))
    missing = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(x), inch(y), inch(w), inch(h)
    )
    missing.fill.solid()
    missing.fill.fore_color.rgb = RGBColor(226, 232, 240)
    missing.line.color.rgb = BORDER
    add_text(slide, f"Missing asset:\n{path.name}", x + 0.2, y + 0.2, w - 0.4, h - 0.4, 12, True, MUTED, PP_ALIGN.CENTER)
    return missing


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

    sw = prs.slide_width / 914400
    sh = prs.slide_height / 914400
    clear_body_only(slide, sh)

    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = LIGHT

    # Body background. Template header/footer artwork remains untouched.
    body_top = 2.45
    body_bottom = 41.18
    body_h = body_bottom - body_top
    bg_rect = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, inch(0), inch(body_top), inch(sw), inch(body_h)
    )
    bg_rect.fill.solid()
    bg_rect.fill.fore_color.rgb = RGBColor(247, 251, 252)
    bg_rect.line.fill.background()

    # Title block.
    add_text(slide, "Audio-Centric Visual Speech Reconstruction", 1.05, 2.88, 31.0, 0.72, 34, True, NAVY, PP_ALIGN.CENTER)
    add_text(
        slide,
        "Lip and audio latents carry the speech signal; text is a weak alignment cue",
        2.1,
        3.72,
        28.9,
        0.42,
        16,
        True,
        TEAL,
        PP_ALIGN.CENTER,
    )
    claim = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, inch(2.2), inch(4.45), inch(28.7), inch(1.1)
    )
    claim.fill.solid()
    claim.fill.fore_color.rgb = NAVY
    claim.line.fill.background()
    claim.adjustments[0] = 0.08
    add_text(
        slide,
        "Replace vision -> text -> TTS with direct Mimi latent reconstruction",
        2.45,
        4.73,
        28.2,
        0.42,
        21,
        True,
        WHITE,
        PP_ALIGN.CENTER,
    )

    # Central architecture.
    add_panel(slide, "1  SYSTEM ARCHITECTURE", 1.05, 6.15, 31.0, 14.65)
    add_image(slide, ASSET_DIR / "generated_architecture_white.png", 1.55, 6.75, 30.0, 12.75)
    add_text(
        slide,
        "Generated architecture visual, with exact interpretation: lip video and weak text/timbre conditions drive a residual Mimi latent reconstructor and decoder.",
        2.0,
        19.65,
        29.0,
        0.48,
        11,
        False,
        MUTED,
        PP_ALIGN.CENTER,
    )

    # Method, evidence, and data row.
    row_y = 21.25
    gap = 0.35
    col_w = (31.0 - 2 * gap) / 3
    x1 = 1.05
    x2 = x1 + col_w + gap
    x3 = x2 + col_w + gap

    add_panel(slide, "2  DATA / CONDITIONS", x1, row_y, col_w, 8.1)
    add_condition_table(slide, x1 + 0.28, row_y + 0.72, col_w - 0.56, 4.8)
    add_text(
        slide,
        "The final path uses AVSR-compatible lip crops rather than the old RGB lip crop. External videos use lip-AVSR text with uniform alignment.",
        x1 + 0.32,
        row_y + 5.95,
        col_w - 0.64,
        1.2,
        11,
        False,
        SLATE,
    )
    add_tag(slide, "text is support, not bottleneck", x1 + 0.75, row_y + 7.25, col_w - 1.5, TEAL)

    add_panel(slide, "3  DETERMINISTIC RECON", x2, row_y, col_w, 8.1)
    add_image(slide, ASSET_DIR / "generated_residual_method_white.png", x2 + 0.3, row_y + 0.82, col_w - 0.6, 3.1)
    add_text(slide, "C = {visual, text, speaker, audio prompt, timbre}", x2 + 0.38, row_y + 4.25, col_w - 0.76, 0.35, 12, True, TEAL, PP_ALIGN.CENTER)
    add_text(slide, "y_hat = y_base + delta", x2 + 0.38, row_y + 4.82, col_w - 0.76, 0.45, 18, True, GREEN, PP_ALIGN.CENTER)
    add_text(
        slide,
        "The final checkpoint fixes x_tilde = 0 and tau = 1. It is endpoint regression through a frozen base plus residual correction, not an iterative FM sampling loop.",
        x2 + 0.38,
        row_y + 5.55,
        col_w - 0.76,
        1.25,
        11,
        False,
        SLATE,
    )

    add_panel(slide, "4  EVIDENCE", x3, row_y, col_w, 8.1)
    add_bar_chart(slide, x3 + 0.65, row_y + 1.05, col_w - 1.3, 3.0)
    add_text(slide, "Final full-val metrics, normalized Mimi latent space", x3 + 0.35, row_y + 4.55, col_w - 0.7, 0.28, 10, True, MUTED, PP_ALIGN.CENTER)
    rows = [
        ("final corr", "0.5819"),
        ("GT -> AVSR text drop", "0.0035"),
        ("FM sampling corr", "~0.35"),
    ]
    for i, (label, value) in enumerate(rows):
        y = row_y + 5.05 + i * 0.55
        add_text(slide, label, x3 + 0.65, y, col_w - 2.2, 0.25, 11, True, NAVY)
        add_text(slide, value, x3 + col_w - 1.85, y, 1.2, 0.25, 11, True, TEAL, PP_ALIGN.RIGHT)
    add_text(
        slide,
        "Timbre/audio-prompt conditioning and residual endpoint recon account for the main gains; exact transcript quality is not the dominant failure mode.",
        x3 + 0.4,
        row_y + 6.85,
        col_w - 0.8,
        0.82,
        10,
        False,
        SLATE,
    )

    # Demo band.
    demo_y = 30.05
    add_panel(slide, "5  SILENT-REFERENCE DEMO", 1.05, demo_y, 31.0, 5.35)
    add_image(slide, ASSET_DIR / "generated_silent_demo_white.png", 1.55, demo_y + 0.72, 15.2, 3.35)
    thumb_w = 3.05
    add_captioned_image(slide, ASSET_DIR / "trump_silent_frame.png", "silent input", 17.3, demo_y + 0.86, thumb_w, thumb_w)
    add_captioned_image(slide, ASSET_DIR / "trump_ref_wave.png", "reference waveform", 20.65, demo_y + 0.86, 4.2, thumb_w)
    add_captioned_image(slide, ASSET_DIR / "trump_generated_frame.png", "generated crop", 25.2, demo_y + 0.86, thumb_w, thumb_w)
    add_text(
        slide,
        "Trump example: silent MP4 + short reference segment -> voiced MP4. The visual stream is preserved; only the speech track is reconstructed.",
        17.25,
        demo_y + 4.24,
        13.95,
        0.42,
        11,
        True,
        NAVY,
        PP_ALIGN.CENTER,
    )

    # Bottom takeaway row.
    take_y = 36.0
    add_panel(slide, "6  TAKEAWAY", 1.05, take_y, 15.15, 4.65)
    add_text(
        slide,
        "We should not treat lip video as a text-recognition problem followed by speech synthesis.",
        1.45,
        take_y + 0.9,
        14.35,
        0.6,
        17,
        True,
        TEAL,
    )
    add_bullets(
        slide,
        [
            "Lip motion provides phonetic and timing evidence.",
            "Mimi latents preserve codec-compatible acoustic structure.",
            "Weak text stabilizes content but does not need to be perfect.",
            "Timbre/reference conditions carry speaker and recording color.",
        ],
        1.45,
        take_y + 1.75,
        14.35,
        2.1,
        12,
    )
    add_text(
        slide,
        "This supports an audio-latent reconstruction architecture over a strict text cascade.",
        1.45,
        take_y + 3.72,
        14.3,
        0.35,
        11,
        True,
        NAVY,
    )

    add_panel(slide, "7  ARTIFACTS", 16.9, take_y, 15.15, 4.65)
    add_bullets(
        slide,
        [
            "scripts/run_raw_video_avsr_recon_pipeline.py",
            "scripts/gradio_avsr_gui.py",
            "poster/fm_avsr_poster.pptx and PDF preview",
            "data/assets/trump_silent_ref_demo/*.mp4",
            "eval_out/trump_raw_prompt_pipeline/.../avsr_text_lipavsr.txt",
        ],
        17.3,
        take_y + 0.9,
        14.4,
        2.75,
        12,
    )
    add_text(
        slide,
        "Generated visual components are stored under poster/assets/generated_*.png.",
        17.3,
        take_y + 3.45,
        14.35,
        0.42,
        10,
        False,
        MUTED,
    )

    prs.save(str(OUT))
    print(OUT)


if __name__ == "__main__":
    build()
