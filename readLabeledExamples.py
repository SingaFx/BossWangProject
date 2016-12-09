import os
import glob
import numpy as np
from PIL import Image

def crop_and_scale_image(im):
    """ Crops and scales a given image. 
        Args: 
            im (PIL Image) : image object to be cropped and scaled
        Returns: 
            (PIL Image) : cropped and scaled image object
    """
    w, h = im.size
    if w > h:
        diff = w - h
        left = int(np.floor(diff / 2.0))
        right = w - (diff - left)
        im_crop = im.crop((left, 0, right, h))
    else:
        diff = h - w
        top = int(np.floor(diff / 2.0))
        bottom = h - (diff - top)
        im_crop = im.crop((0, top, w, bottom))
    return im_crop.resize((100, 100), Image.ANTIALIAS)

w, h = 28, 28
labels = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
char_to_class = {labels[i]:i for i in xrange(0, len(labels))}

data_dir = "data/letter/"
images = []
labels = []
for dir_name in os.listdir(data_dir):
	if dir_name not in char_to_class:
		continue
	label = char_to_class[dir_name]
	subdir = data_dir + dir_name + '/'
	for file in glob.glob(subdir + "*.jpeg"):
		im = Image.open(file)
		im_expanded = im
		im_scaled = im_expanded.resize((w, h), Image.ANTIALIAS)
		images.append(np.asarray(im_scaled).reshape(w * h, 1))
		labels.append(label)

num_examples = len(images)
dev_pixels = np.zeros((num_examples, w * h), dtype=np.ubyte)
dev_labels = np.zeros(num_examples, dtype=np.ubyte)

for i in xrange(0, num_examples):
	dev_pixels[i, :] = images[i][:, 0]
	dev_labels[i] = labels[i]

np.save("dev-pixels.ubyte", dev_pixels)
np.save("dev-labels.ubyte", dev_labels)




