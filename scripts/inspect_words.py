import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")

for page_idx, page in enumerate(doc):
    print(f"\n=================== PAGE {page_idx + 1} ===================")
    words = page.get_text("words")
    collisions = 0
    for i in range(len(words)):
        w1 = words[i]
        for j in range(i + 1, min(i + 100, len(words))):
            w2 = words[j]
            # If from different blocks or lines
            if w1[5] != w2[5] or w1[6] != w2[6]:
                r1 = fitz.Rect(w1[:4])
                r2 = fitz.Rect(w2[:4])
                inter = r1.intersect(r2)
                if not inter.is_empty and inter.get_area() > 8:
                    print(f"P{page_idx+1} Word collision: '{w1[4]}' (blk {w1[5]}) and '{w2[4]}' (blk {w2[5]}) at ({r1.x0:.1f}, {r1.y0:.1f})")
                    collisions += 1
    if collisions == 0:
        print(f"P{page_idx+1}: No word collisions detected.")
