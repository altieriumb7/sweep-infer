import fitz # PyMuPDF
import os

doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")
os.makedirs(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages", exist_ok=True)

print("Page count:", len(doc))
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    out_path = f"C:\\Users\\Umberto\\synaptic-fold-v2\\pdf_pages\\page_{i+1}.png"
    pix.save(out_path)
    print(f"Saved {out_path} ({pix.width}x{pix.height})")
