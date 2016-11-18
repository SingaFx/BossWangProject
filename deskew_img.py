import cv2

import numpy as np
import math
import os

def houghSpace(im):
    maxTheta = 180
    #the width of the space corresponds to the maximum angle taken into account
    houghMatrixCols = maxTheta

    #original image size
    h, w = im.shape
    #cannot exist in the image more than its diagonal
    rhoMax = math.sqrt(w * w + h * h)

    #the height of the space is twice the maximum rho, to also consider the negative rho
    houghMatrixRows = int(rhoMax) * 2 + 1
    #the calculated rho will be shifted by half of the height to be able to be in space, both the negative rho
    rhoOffset = houghMatrixRows/2

    degToRadScale = math.pi / 180
    rangemaxTheta = range(maxTheta)
    sin, cos = zip(*((math.sin(i * degToRadScale), math.cos(i * degToRadScale)) for i in rangemaxTheta))

    #initialization of the space
    houghSpace = [0.0 for x in xrange(houghMatrixRows * houghMatrixCols)]

    #skim across the original image
    for y in xrange(h):
        for x in xrange(w):
            #for each edge point
            if im[y, x] > 0:
                #calculating his bundle of straight lines...
                for theta in rangemaxTheta:
                    # for each theta angle in space, calculate the relative value of rho
                    # using the polar form of the equation of the straight line
                    rho = int(round(x * cos[theta] + y * sin[theta] + rhoOffset))

                    # Once known the coordinates theta and rho, increment the counter of the Hough space to the coordinate
                    c = rho * houghMatrixCols + theta
                    houghSpace[c] = houghSpace[c] + 1

    houghSpace = houghSpace / np.max(houghSpace)
    return np.reshape(houghSpace , (houghMatrixRows, houghMatrixCols))

def deskew(filename, directory):
    outdir = directory + '/rotate/'
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    h, w = im.shape

    # image resizing
    im = cv2.resize(im, (w/3,h/3))

    # canny edge detector
    im = cv2.Canny(im, 100, 200)

    # Application of the calculation of the space of the image input hough
    hSpace = houghSpace(im)

    # filtering of peaks
    hSpace[hSpace < 0.8] = 0
    # histogram calculation
    hist = sum(hSpace)
    # perpendicular angle calculation
    theta1 = 90 - np.argmax(hist)


    # color image reading
    im = cv2.imread(filename, cv2.IMREAD_COLOR)

    # image rotation
    h, w, d = im.shape
    rotation_M = cv2.getRotationMatrix2D((w / 2, h / 2), -theta1, 1)
    rotated_im = cv2.warpAffine(im, rotation_M, (w,h), flags=cv2.INTER_CUBIC,borderValue=(255,255,255))

    # write deskew image
    print outdir + filename
    cv2.imwrite(outdir + filename, rotated_im)



if __name__ == '__main__':
    directory = os.getcwd()
    indir = directory + '/output/'
    os.chdir(indir)
    outdir = directory + '/rotate/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for filename in os.listdir(indir):
        if filename.endswith(".jpeg"):
            deskew(filename, directory)

