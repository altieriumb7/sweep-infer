import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")
page3 = doc[2]

# Words inside Table 2 region: x in [305, 565], y in [290, 430]
words = [w for w in page3.get_text("words") if 305 <= w[0] <= 565 and 290 <= w[1] <= 430]

# Group words by line (similar y0)
lines = {}
for w in words:
    # cluster by y0 within 2 points
    matched = False
    for y_key in lines:
        if abs(w[1] - y_key) < 3:
            lines[y_key].append(w)
            matched = True
            break
    if not matched:
        lines[w[1]] = [w]

for y_key in sorted(lines.keys()):
    line_words = sorted(lines[y_key], key=lambda x: x[0])
    text = " ".join([w[4] for w in line_words])
    coords = " | ".join([f"{w[4]} ({w[0]:.1f}-{w[2]:.1f})" for w in line_words])
    print(f"y={y_key:.1f}: {text}")
    # check if any two words in line overlap in x
    for k in range(len(line_words) - 1):
        if line_words[k][2] > line_words[k+1][0]:
            print(f"   OVERLAP in Table 2: '{line_words[k][4]}' (ends {line_words[k][2]:.1f}) overlaps '{line_words[k+1][4]}' (starts {line_words[k+1][0]:.1f})")
