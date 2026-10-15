"""Renderiza um vídeo vertical (1080x1920) animado a partir de um áudio.

Uso: python3 render_video.py assets/audio.m4a saida.mp4
Requer: ffmpeg, numpy, pillow.
"""
import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 48000
TOP_LINE = "BAR E RESTAURANTE"
NAME_1 = "BIG"
NAME_2 = "BROTHER"
TAGLINE = "Comida boa • Bebida gelada • Música"
CTA = "VENHA NOS VISITAR!"

GOLD = (255, 190, 60)
ORANGE = (255, 110, 40)
RED = (230, 40, 60)
WHITE = (255, 255, 255)

FONT_BLACK = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
FONT_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"


def load_audio(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def analyse(samples, n_frames):
    """Espectro (64 bandas) e energia de graves por frame, suavizados e normalizados."""
    win = 4096
    hop = SR / FPS
    padded = np.concatenate([np.zeros(win), samples, np.zeros(win)])
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1 / SR)
    edges = np.geomspace(60, 9000, 65)
    bands = np.zeros((n_frames, 64))
    bass = np.zeros(n_frames)
    level = np.zeros(n_frames)
    for i in range(n_frames):
        c = int(i * hop) + win
        seg = padded[c - win // 2:c + win // 2] * window
        mag = np.abs(np.fft.rfft(seg))
        for b in range(64):
            m = (freqs >= edges[b]) & (freqs < edges[b + 1])
            bands[i, b] = mag[m].mean() if m.any() else 0
        bass[i] = mag[(freqs > 40) & (freqs < 250)].mean()
        level[i] = np.sqrt(np.mean(seg ** 2))
    bands = np.log1p(bands * 4)
    bands /= np.percentile(bands, 99) + 1e-9
    bass = np.log1p(bass * 4)
    bass /= np.percentile(bass, 99) + 1e-9
    level /= np.percentile(level, 99) + 1e-9
    # suavização: sobe rápido, desce devagar
    for arr in (bands, bass, level):
        for i in range(1, n_frames):
            arr[i] = np.maximum(arr[i], arr[i - 1] * 0.82)
    return np.clip(bands, 0, 1), np.clip(bass, 0, 1), np.clip(level, 0, 1)


def ease_out_back(t):
    t = min(max(t, 0), 1)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out(t):
    t = min(max(t, 0), 1)
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3))


def make_background():
    y, x = np.mgrid[0:H, 0:W]
    d = np.sqrt((x - W / 2) ** 2 + (y - H * 0.42) ** 2) / (H * 0.7)
    d = np.clip(d, 0, 1)[..., None]
    inner = np.array([70, 22, 10], dtype=float)
    outer = np.array([8, 4, 10], dtype=float)
    img = inner * (1 - d) + outer * d
    return Image.fromarray(img.astype(np.uint8), "RGB")


def text_center(draw, y, text, font, fill, spacing=0):
    if spacing:
        widths = [draw.textlength(ch, font=font) for ch in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = (W - total) / 2
        for ch, w in zip(text, widths):
            draw.text((x, y), ch, font=font, fill=fill)
            x += w + spacing
    else:
        w = draw.textlength(text, font=font)
        draw.text(((W - w) / 2, y), text, font=font, fill=fill)


def main(audio_path, out_path):
    samples = load_audio(audio_path)
    duration = len(samples) / SR
    n_frames = int(duration * FPS)
    bands, bass, level = analyse(samples, n_frames)

    bg = make_background()
    rng = np.random.default_rng(7)
    particles = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(3, 10),
                  rng.uniform(20, 70), rng.uniform(0, 2 * math.pi)) for _ in range(70)]

    f_top = ImageFont.truetype(FONT_BOLD, 50)
    f_big = ImageFont.truetype(FONT_BLACK, 210)
    f_bro = ImageFont.truetype(FONT_BLACK, 130)
    f_tag = ImageFont.truetype(FONT_SEMI, 40)
    f_cta = ImageFont.truetype(FONT_BLACK, 74)

    cx, cy = W / 2, H * 0.42
    cta_start = max(duration - 5.0, duration * 0.7)

    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio_path,
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)

    for i in range(n_frames):
        t = i / FPS
        b = float(bass[i])
        spec = bands[i]
        frame = bg.copy()

        # brilho pulsante atrás do logo
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gr = 330 + b * 120
        gd.ellipse([cx - gr, cy - gr, cx + gr, cy + gr], fill=lerp((120, 40, 0), (255, 120, 20), b))
        glow = glow.filter(ImageFilter.GaussianBlur(110))
        frame = Image.fromarray(np.clip(np.asarray(frame, dtype=np.int16)
                                        + np.asarray(glow, dtype=np.int16), 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(frame)

        # partículas (bokeh) subindo
        for (px, py, r, sp, ph) in particles:
            yy = (py - t * sp * (1 + b)) % H
            xx = px + math.sin(t * 1.2 + ph) * 25
            a = 0.35 + 0.65 * abs(math.sin(t * 2 + ph))
            col = lerp((40, 15, 5), GOLD, a * (0.5 + 0.5 * b))
            d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=col)

        # anel de espectro circular (espelhado)
        intro = ease_out_back(t / 0.9)
        base_r = (300 + b * 35) * max(intro, 0.01)
        n = 128
        for k in range(n):
            idx = k if k < n // 2 else n - 1 - k
            v = float(spec[min(int(idx * 64 / (n // 2)), 63)])
            ang = k / n * 2 * math.pi - math.pi / 2
            ln = (14 + v * 210) * ease_out(t / 1.2)
            r0 = base_r + 22
            x0, y0 = cx + math.cos(ang) * r0, cy + math.sin(ang) * r0
            x1, y1 = cx + math.cos(ang) * (r0 + ln), cy + math.sin(ang) * (r0 + ln)
            d.line([x0, y0, x1, y1], fill=lerp(ORANGE, GOLD, v), width=9)

        # disco central
        if base_r > 5:
            d.ellipse([cx - base_r - 8, cy - base_r - 8, cx + base_r + 8, cy + base_r + 8], fill=GOLD)
            d.ellipse([cx - base_r, cy - base_r, cx + base_r, cy + base_r], fill=(20, 10, 12))
            spin = t * 0.8
            for k in range(3):
                rr = base_r * (0.93 - k * 0.04)
                d.arc([cx - rr, cy - rr, cx + rr, cy + rr],
                      start=math.degrees(spin) + k * 120, end=math.degrees(spin) + k * 120 + 70,
                      fill=lerp(RED, ORANGE, k / 2), width=6)

        # nome do restaurante dentro do disco (com "pop" na batida)
        if t > 0.35:
            pop = ease_out_back((t - 0.35) / 0.6)
            scale = pop * (1 + b * 0.06)
            layer = Image.new("RGBA", (W, 600), (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            w1 = ld.textlength(NAME_1, font=f_big)
            ld.text(((W - w1) / 2 + 6, 40 + 6), NAME_1, font=f_big, fill=(0, 0, 0, 160))
            ld.text(((W - w1) / 2, 40), NAME_1, font=f_big, fill=GOLD)
            w2 = ld.textlength(NAME_2, font=f_bro)
            ld.text(((W - w2) / 2 + 5, 280 + 5), NAME_2, font=f_bro, fill=(0, 0, 0, 160))
            ld.text(((W - w2) / 2, 280), NAME_2, font=f_bro, fill=WHITE)
            if scale > 0.02:
                sw, sh = int(W * scale), int(600 * scale)
                layer = layer.resize((sw, sh), Image.LANCZOS)
                frame.paste(layer, (int(cx - sw / 2), int(cy - sh / 2 - 10 * scale)), layer)
                d = ImageDraw.Draw(frame)

        # linha superior deslizando
        if t > 0.8:
            a = ease_out((t - 0.8) / 0.6)
            y = H * 0.13 - (1 - a) * 80
            text_center(d, y, TOP_LINE, f_top, lerp((20, 10, 10), WHITE, a), spacing=10)
            lw = 360 * a
            d.rectangle([cx - lw, y + 80, cx + lw, y + 86], fill=ORANGE)

        # slogan
        if t > 1.4:
            a = ease_out((t - 1.4) / 0.7)
            text_center(d, H * 0.69 + (1 - a) * 60, TAGLINE, f_tag, lerp((20, 10, 10), (255, 225, 190), a))

        # equalizador inferior
        nb, gap = 40, 8
        bw = (W - 160 - gap * (nb - 1)) / nb
        base_y = H * 0.86
        for k in range(nb):
            v = float(spec[int(k * 64 / nb)])
            hh = 8 + v * 150 * ease_out(t / 1.0)
            x = 80 + k * (bw + gap)
            d.rounded_rectangle([x, base_y - hh, x + bw, base_y], radius=6, fill=lerp(RED, GOLD, v))
            d.rounded_rectangle([x, base_y + 8, x + bw, base_y + 8 + hh * 0.3], radius=6,
                                fill=lerp((40, 12, 10), (110, 70, 20), v))

        # chamada final
        if t > cta_start:
            a = ease_out_back((t - cta_start) / 0.6)
            pulse = 1 + 0.04 * math.sin(t * 8) + b * 0.04
            bw2, bh2 = 760 * a * pulse, 130 * a * pulse
            by = H * 0.76
            if bw2 > 10:
                d.rounded_rectangle([cx - bw2 / 2, by - bh2 / 2, cx + bw2 / 2, by + bh2 / 2],
                                    radius=int(bh2 / 2), fill=RED)
                if a > 0.6:
                    tw = d.textlength(CTA, font=f_cta)
                    d.text((cx - tw / 2, by - 48), CTA, font=f_cta, fill=WHITE)

        # fade in / fade out
        fade = min(1.0, t / 0.4, (duration - t) / 0.5)
        if fade < 1:
            frame = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), frame, max(fade, 0))

        ff.stdin.write(frame.tobytes())
        if i % 60 == 0:
            print(f"{i}/{n_frames}", flush=True)

    ff.stdin.close()
    ff.wait()
    print("ok:", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
