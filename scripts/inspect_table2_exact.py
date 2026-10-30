import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")
page3 = doc[2]
words = [w for w in page3.get_text("words") if 305 <= w[0] <= 565 and 305 <= w[1] <= 385]

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
    print(f"--- Row at y={yk:.1f} ---")
    for w in row:
        print(f"  {w[4]:<20} x=[{w[0]:.2f}, {w[2]:.2f}] width={w[2]-w[0]:.2f}")
