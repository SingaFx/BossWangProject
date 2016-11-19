from PIL import Image, ImageDraw, ImageFont
import string
import os
import numpy as np
import cv2
# create directory

for txt in string.digits + string.ascii_uppercase:
    if txt == "O":
        continue
    outdir = "generate/{0}/".format(txt)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

w, h = 150, 150
# x, y(tranlation), fontsize(scale), theta (rotation)
options = []
for x in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
    for y in [-12, -6, 0, 6, 12]:
        for fontsize in [120, 140, 160, 180]:
            for theta in [-10, -5, 0, 5, 10]:
                if fontsize == 180 and (theta == 10 or theta == -10):
                    continue
                for font_file in ["Highway Gothic.ttf", "Filetto_regular.ttf"]:
                    options.append((x, y, fontsize, theta, font_file))

for x, y, fontsize, theta, font_file in options:
    for txt in string.digits:
        folder = txt
        name = str(np.random.randint(0, 999999999))
        image = Image.new("RGBA", (w, h), (255,255,255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_file, fontsize)
        draw.text((x, y), txt, (0,0,0), font=font)
        rotated_im = image.rotate(theta).crop((10, 10, 140, 140))
        img_resized = rotated_im.resize((28,28), Image.ANTIALIAS)
        img_resized.save("generate/{0}/{1}.jpeg".format(folder, name))

    for txt in string.ascii_uppercase:
        folder = "0" if txt == "O" else txt
        folder = "1" if txt == "I" else folder
        name = str(np.random.randint(0, 999999999))
        image = Image.new("RGBA", (w, h), (255,255,255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_file, fontsize)
        draw.text((x, y), txt, (0,0,0), font=font)
        rotated_im = image.rotate(theta).crop((10, 10, 140, 140))
        img_resized = rotated_im.resize((28,28), Image.ANTIALIAS)
        img_resized.save("generate/{0}/{1}.jpeg".format(folder, name))
    
    for txt in string.ascii_lowercase:
        folder = "0" if txt == "o" else txt
        folder = "1" if txt == "l" else folder
        name = str(np.random.randint(0, 999999999))
        image = Image.new("RGBA", (w, h), (255,255,255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_file, fontsize)
        draw.text((x, y), txt, (0,0,0), font=font)
        rotated_im = image.rotate(theta).crop((10, 10, 140, 140))
        img_resized = rotated_im.resize((28,28), Image.ANTIALIAS)
        img_resized.save("generate/{0}/{1}.jpeg".format(folder, name))

# from captcha.image import ImageCaptcha

# image = ImageCaptcha(fonts=['Highway Gothic.ttf'])

# data = image.generate('1')
# image.write('1', 'out.png')