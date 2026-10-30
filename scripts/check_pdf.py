import sys

for mod in ["fitz", "pypdf", "pdf2image", "pdfplumber", "pypdf2"]:
    try:
        __import__(mod)
        print(mod, ": available")
    except Exception as e:
        print(mod, ": not available", e)
