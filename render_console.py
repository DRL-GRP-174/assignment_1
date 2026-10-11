"""Turns a text log into a terminal-style PNG for the slides.
usage: python render_console.py <log.txt> <out.png> "<command shown on first line>" [max_lines]"""
import sys

from PIL import Image, ImageDraw, ImageFont

log, out, cmd = sys.argv[1], sys.argv[2], sys.argv[3]
max_lines = int(sys.argv[4]) if len(sys.argv) > 4 else 14
lines = [l.rstrip("\n") for l in open(log, encoding="utf-8") if l.strip()][-max_lines:]

try:
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 26)
except OSError:
    font = ImageFont.load_default()

pad, lh = 28, 36
width = max(font.getlength(l) for l in lines + ["PS> " + cmd]) + 2 * pad
height = 54 + pad + lh * (len(lines) + 1) + pad
img = Image.new("RGB", (int(width), int(height)), "#1E1E1E")
d = ImageDraw.Draw(img)
d.rectangle([0, 0, width, 40], fill="#323233")
d.text((18, 6), "Windows PowerShell  -  ppo_group174", fill="#BBBBBB", font=font)

y = 54 + pad // 2
d.text((pad, y), "PS> ", fill="#4EC9B0", font=font)
d.text((pad + font.getlength("PS> "), y), cmd, fill="#DCDCAA", font=font)
for l in lines:
    y += lh
    d.text((pad, y), l, fill="#D4D4D4", font=font)
img.save(out)
print("saved", out, img.size)
