"""README 비교 그림: 작은 사진을 바이큐빅으로 키운 것 | SeedVR2 결과 | 줄이기 전 원본 (글자 없음).

python compare.py <작은 사진> <SeedVR2 결과> <원본> <출력.jpg> [배율 0.75]
"""
import sys

from PIL import Image

small, result, full, dest = sys.argv[1:5]
scale = float(sys.argv[5]) if len(sys.argv) > 5 else 0.75
out = Image.open(result).convert("RGB")
w, h = out.size
panels = [
    Image.open(small).convert("RGB").resize((w, h), Image.BICUBIC),
    out,
    Image.open(full).convert("RGB").resize((w, h), Image.LANCZOS),
]
canvas = Image.new("RGB", (w * 3 + 32, h), (236, 238, 241))
for i, p in enumerate(panels):
    canvas.paste(p, (i * (w + 16), 0))
canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.LANCZOS)
canvas.save(dest, quality=90)
print("ok", dest, canvas.size)
