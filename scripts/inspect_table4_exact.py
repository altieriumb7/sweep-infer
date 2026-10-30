import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")
page6 = doc[5]

# Words inside Table 4 region
words = [w for w in page6.get_text("words") if 40 <= w[0] <= 570 and 40 <= w[1] <= 260]

lines = {}
for w in words:
    matched = False
    for yk in lines:
        if abs(w[1] - yk) < 3:
            lines[yk].append(w)
            matched = True
            break
    if not matched:
        lines[w[1]] = [w]

for yk in sorted(lines.keys()):
    row = sorted(lines[yk], key=lambda x: x[0])
    text = " ".join([w[4] for w in row])
    print(f"y={yk:.1f}: {text[:100]}")
    for k in range(len(row) - 1):
        if row[k][2] > row[k+1][0]:
            print(f"   OVERLAP in Table 4: '{row[k][4]}' overlaps '{row[k+1][4]}'")
