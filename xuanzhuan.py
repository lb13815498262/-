from PIL import Image

img = Image.open('lena.png')
img2 = img.rotate(45)       # 逆时针旋转45°
img2.save("lena_rot45.png")
img2.show()

