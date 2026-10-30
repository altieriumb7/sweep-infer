import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")

page3 = doc[2] # 0-indexed -> Page 3
print("=== PAGE 3 TEXT BLOCKS ===")
for b in page3.get_text("blocks"):
    x0, y0, x1, y1, text, block_no, block_type = b
    if text.strip():
        first_line = text.strip().split('\n')[0]
        print(f"Block {block_no} ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}): {first_line[:70]}")
