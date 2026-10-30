from PIL import Image

# Page 3: Tables 1, 2, 3
im3 = Image.open(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_3.png")
# Table 1 is on the left bottom
# Table 2 is on the right middle
# Table 3 is on the right bottom

# Bounding boxes on 1275x1650 (scale factor = 1275 / 612 = 2.0833):
# Table 1: x in [53, 303], y in [580, 730] in points -> * 2.0833
t1_box = (int(50 * 2.0833), int(570 * 2.0833), int(310 * 2.0833), int(740 * 2.0833))
im3.crop(t1_box).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\crop_table1.png")

# Table 2: x in [310, 560], y in [300, 420] in points
t2_box = (int(305 * 2.0833), int(290 * 2.0833), int(565 * 2.0833), int(430 * 2.0833))
im3.crop(t2_box).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\crop_table2.png")

# Table 3: x in [310, 560], y in [600, 730]
t3_box = (int(305 * 2.0833), int(600 * 2.0833), int(565 * 2.0833), int(740 * 2.0833))
im3.crop(t3_box).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\crop_table3.png")

# Page 4: Figures 1, 2, 3, 4
im4 = Image.open(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_4.png")
# Fig 1 (top left), Fig 2 (top right), Fig 3 (bottom left), Fig 4 (bottom right)
f1_box = (int(40 * 2.0833), int(40 * 2.0833), int(310 * 2.0833), int(200 * 2.0833))
im4.crop(f1_box).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\crop_fig1.png")

f2_box = (int(305 * 2.0833), int(40 * 2.0833), int(570 * 2.0833), int(200 * 2.0833))
im4.crop(f2_box).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\crop_fig2.png")

f3_box = (int(40 * 2.0833), int(200 * 2.0833), int(310 * 2.0833), int(360 * 2.0833))
im4.crop(f3_box).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\crop_fig3.png")

f4_box = (int(305 * 2.0833), int(200 * 2.0833), int(570 * 2.0833), int(360 * 2.0833))
im4.crop(f4_box).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\crop_fig4.png")

# Page 6: Table 4
im6 = Image.open(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\page_6.png")
t4_box = (int(40 * 2.0833), int(40 * 2.0833), int(570 * 2.0833), int(250 * 2.0833))
im6.crop(t4_box).save(r"C:\Users\Umberto\synaptic-fold-v2\pdf_pages\crop_table4.png")

print("All crops saved successfully.")
