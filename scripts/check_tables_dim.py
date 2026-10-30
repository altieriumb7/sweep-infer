from PIL import Image

for tname in ["crop_table1.png", "crop_table2.png", "crop_table3.png", "crop_table4.png"]:
    im = Image.open(rf"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\{tname}")
    print(f"{tname}: size = {im.size}")
