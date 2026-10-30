from PIL import Image

# Let's crop Page 3 into left and right, or inspect Table 1, 2, 3 on Page 3
im3 = Image.open(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_3.png")
# Image size is 1275 x 1650
# Let's save top half, bottom half, etc.
im3.crop((0, 0, 1275, 825)).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_3_top.png")
im3.crop((0, 825, 1275, 1650)).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_3_bottom.png")

# Page 4
im4 = Image.open(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_4.png")
im4.crop((0, 0, 1275, 825)).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_4_top.png")
im4.crop((0, 825, 1275, 1650)).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_4_bottom.png")

# Page 6
im6 = Image.open(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_6.png")
im6.crop((0, 0, 1275, 825)).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_6_top.png")

print("Crops saved.")
