import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")

page2 = doc[1]
print("=== PAGE 2 BLOCKS ===")
for i, b in enumerate(page2.get_text("blocks")):
    x0, y0, x1, y1, text, block_no, block_type = b
    if text.strip():
        print(f"Block {i} (no {block_no}) ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}):")
        print(repr(text[:120]))
