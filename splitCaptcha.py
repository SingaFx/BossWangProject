import numpy as np
import cv2
import os


def splitReCaptcha(im, directory):
    img = cv2.imread(im, 0)
    w = img.shape[1]
    sp = w/6
    while not np.all((255 - img[:, sp]) <= 10):
        sp += 3
    left = img[:, :sp]
    right = img[:, sp:]

    outdir = directory + '/output/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cv2.imwrite(os.path.join(outdir, 'l_' + im), left)
    cv2.imwrite(os.path.join(outdir, 'r_' + im), right)


if __name__ == '__main__':
    directory = os.getcwd()
    indir = directory + '/input/'
    os.chdir(indir)
    for filename in os.listdir(indir):
        if filename.endswith(".jpeg"):
            try:
                splitReCaptcha(filename, directory)
            except:
                print "Error splitting " , filename
