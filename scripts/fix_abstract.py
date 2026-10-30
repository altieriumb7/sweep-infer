path = r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.tex"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("an ($O(1)$) zero-penalty", "an $O(1)$ zero-penalty")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Fixed O(1)")
