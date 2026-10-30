from PIL import Image

# Let's inspect page 2, 3, 4, 5
# Each image is 1275 x 1650
# Let's check where tables are on each page

import fitz

doc = fitz.open(r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.pdf")

for p_num, page in enumerate(doc):
    text = page.get_text()
    for tab_name in ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5"]:
        if tab_name.lower() in text.lower():
            print(f"{tab_name} is on Page {p_num + 1}")
    for fig_name in ["Figure 1", "Figure 2", "Figure 3", "Figure 4"]:
        if fig_name.lower() in text.lower():
            print(f"{fig_name} is on Page {p_num + 1}")
