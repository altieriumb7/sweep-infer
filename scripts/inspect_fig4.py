import fitz
import sys

sys.stdout.reconfigure(encoding='utf-8')
doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\figures\fig4_cross_platform_spectrum.pdf")
print("=== FIG 4 TEXT ===")
for b in doc[0].get_text("blocks"):
    print(b[:5])
