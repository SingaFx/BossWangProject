from PIL import Image, ImageDraw, ImageFont
import string
import os
import numpy as np
import cv2
# create directory

final_h, final_w = 28, 28
labels = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
char_to_class = {labels[i]:i for i in xrange(0, len(labels))}

def drawExample(name, folder, option, save=True):
    x, y, fontsize, theta, font_file = option
    image = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_file, fontsize)
    draw.text((x, y), txt, 0, font=font)
    rotated_im = image.rotate(theta).crop((10, 10, 140, 140))
    img_resized = rotated_im.resize((final_h, final_w), Image.ANTIALIAS)
    if save:
        img_resized.save("generate/{0}/{1}.jpeg".format(folder, name))
    return img_resized

def map_char_to_class(char):
    return char_to_class[char.upper()]

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
num_examples = len(options) * (10 + 26 * 2)
num_pixels = final_h * final_w
pixels = np.zeros((num_examples, num_pixels), dtype=np.ubyte)
labels = np.zeros(num_examples, dtype=np.ubyte)
print num_examples
k = 0
for option in options:
    for txt in string.digits:
        folder = txt
        name = str(np.random.randint(0, 999999999))
        img_resized = drawExample(name, folder, option)
        array = np.array(img_resized)
        pixels[k, :] = array.reshape(1, num_pixels)
        labels[k] = map_char_to_class(folder)
        k += 1

    for txt in string.ascii_uppercase:
        folder = "0" if txt == "O" else txt
        folder = "1" if txt == "I" else folder
        name = str(np.random.randint(0, 999999999))
        img_resized = drawExample(name, folder, option)
        array = np.array(img_resized)
        pixels[k, :] = array.reshape(1, num_pixels)
        labels[k] = map_char_to_class(folder)
        k += 1
    
    for txt in string.ascii_lowercase:
        folder = "0" if txt == "o" else txt
        folder = "1" if txt == "l" else folder
        name = str(np.random.randint(0, 999999999))
        img_resized = drawExample(name, folder, option)
        array = np.array(img_resized)
        pixels[k, :] = array.reshape(1, num_pixels)
        labels[k] = map_char_to_class(folder)
        k += 1

np.save("pixels.ubyte", pixels)
np.save("labels.ubyte", labels)
# from captcha.image import ImageCaptcha

# image = ImageCaptcha(fonts=['Highway Gothic.ttf'])

# data = image.generate('1')
# image.write('1', 'out.png')