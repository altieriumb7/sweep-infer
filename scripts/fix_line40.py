path = r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.tex"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix nested $O($O(1)$)
assert "$O($O(1)$)" in text, "Target not found!"
text = text.replace("$O($O(1)$)", "$O(1)$")

# Add \setlength{\columnsep}{18pt} right after \documentclass
if "\\setlength{\\columnsep}{18pt}" not in text:
    text = text.replace("\\documentclass[journal,10pt,twocolumn]{article}", "\\documentclass[journal,10pt,twocolumn]{article}\n\\setlength{\\columnsep}{18pt}")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Fixed line 40 and added columnsep!")
