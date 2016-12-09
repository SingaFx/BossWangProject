import numpy as np
import cv2
import os


def splitReCaptcha(im, directory):
    slice_size = 1
    threshold = 25
    img = cv2.imread(im, 0)
    w = img.shape[1]
    sp = 0
    start, end = 0, w-1
    left, right = None, None
    while sp < w and np.all((255 - img[:, sp]) <= threshold):
        sp += slice_size
    start = max(sp - slice_size, start)
    while sp < w and not np.all((255 - img[:, sp]) <= threshold):
        sp += slice_size
    left = sp
    while sp < w and np.all((255 - img[:, sp]) <= threshold):
        sp += slice_size
    right = sp - slice_size
    while sp < w and not np.all((255 - img[:, sp]) <= threshold):
        sp += slice_size
    end = min(end, sp)

    left_img = img[:, start:left]
    right_img = img[:, right:end]

    outdir = directory + '/output/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cv2.imwrite(os.path.join(outdir, 'l_' + im), left_img)
    cv2.imwrite(os.path.join(outdir, 'r_' + im), right_img)


if __name__ == '__main__':
    directory = os.getcwd()
    indir = directory + '/input/'
    os.chdir(indir)
    for filename in os.listdir(indir):
        if filename.endswith(".jpeg"):
            try:
                splitReCaptcha(filename, directory)
            except Exception as e:
                print e
