import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")

for p_num, page in enumerate(doc):
    words = page.get_text("words")
    # Group by line
    lines = {}
    for w in words:
        matched = False
        for y_key in lines:
            if abs(w[1] - y_key) < 2.5 and (w[0] < 305 if lines[y_key][0][0] < 305 else w[0] >= 305):
                lines[y_key].append(w)
                matched = True
                break
        if not matched:
            lines[w[1]] = [w]

    for y_key in sorted(lines.keys()):
        line_words = sorted(lines[y_key], key=lambda x: x[0])
        for k in range(len(line_words) - 1):
            w1 = line_words[k]
            w2 = line_words[k+1]
            # If w1 ends after w2 starts, and both in same column
            if w1[2] > w2[0] + 0.5:
                print(f"P{p_num+1} Line y={y_key:.1f} OVERLAP: '{w1[4]}' ({w1[0]:.1f}-{w1[2]:.1f}) overlaps '{w2[4]}' ({w2[0]:.1f}-{w2[2]:.1f})")
