import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")

print("Checking text layout of all pages...")

for page_idx, page in enumerate(doc):
    # Let's inspect drawings and rects on the page
    drawings = page.get_drawings()
    # Let's check if any text in column 1 (x < 300) crosses over x = 305
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, b_no, b_type = b
        if b_type == 0 and text.strip():
            # If block starts in column 1 and ends past column 2 start (310)
            if x0 < 290 and x1 > 320:
                # Is it full width (x0 < 70 and x1 > 540)?
                if not (x0 < 75 and x1 > 530):
                    print(f"Page {page_idx+1} block crosses between columns: x0={x0:.1f}, x1={x1:.1f}")
                    print(f"   Text: {repr(text[:80])}")
