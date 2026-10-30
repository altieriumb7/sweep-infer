import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")

for page_idx, page in enumerate(doc):
    blocks = page.get_text("blocks")
    print(f"\n=================== PAGE {page_idx + 1} ===================")
    
    text_blocks = [b for b in blocks if b[4].strip() and b[6] == 0]
    
    for i in range(len(text_blocks)):
        b1 = text_blocks[i]
        for j in range(i + 1, len(text_blocks)):
            b2 = text_blocks[j]
            r1 = fitz.Rect(b1[:4])
            r2 = fitz.Rect(b2[:4])
            intersect = r1.intersect(r2)
            if not intersect.is_empty and intersect.get_area() > 30:
                print(f"Overlap on page {page_idx + 1} between block {i} and {j} (area={intersect.get_area():.1f}):")
                print(f"  B1: {r1} -> {repr(b1[4][:60])}")
                print(f"  B2: {r2} -> {repr(b2[4][:60])}")

    mid = page.rect.width / 2
    for idx, b in enumerate(text_blocks):
        x0, y0, x1, y1, text, _, _ = b
        # Full width element starts near left margin (<100) and ends near right margin (>500)
        is_fullwidth = (x0 < 100 and x1 > page.rect.width - 100)
        if not is_fullwidth:
            if x0 < mid - 15 and x1 > mid + 15:
                print(f"Block crossing gutter on page {page_idx + 1} ({x0:.1f}, {x1:.1f}): {repr(text[:80])}")

        # Check if text extends past right margin (> 560)
        if x1 > page.rect.width - 36:
            print(f"Block overflowing right margin on page {page_idx + 1} (x1={x1:.1f}): {repr(text[:80])}")
