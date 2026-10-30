# Let's inspect the visual quality of the cropped images
# We have crop_fig1.png, crop_fig2.png, crop_fig3.png, crop_fig4.png, crop_table1.png, crop_table2.png, crop_table3.png, crop_table4.png
# Let's check their resolutions and file sizes
import os

crops = [
    "crop_fig1.png", "crop_fig2.png", "crop_fig3.png", "crop_fig4.png",
    "crop_table1.png", "crop_table2.png", "crop_table3.png", "crop_table4.png"
]

for c in crops:
    p = os.path.join(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages", c)
    if os.path.exists(p):
        print(f"{c}: size={os.path.getsize(p)} bytes")
