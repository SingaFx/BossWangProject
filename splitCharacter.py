import numpy as np
import cv2
import os
import sys

white = 255
threshold = 240 # white background
diff = 230
num_threshold = 20 # noise
small_threshold = 8
singel_char = 50

def abs_diff(a, b):
    return (a - b) if a > b else (b - a)

def dfs(i, j, prev, img, points):
    h, w = img.shape[0], img.shape[1]
    if i >= 0 and j >= 0 and i < h and j < w and \
        img[i, j] < threshold and abs_diff(img[i, j], prev) < diff:
        points.append([i, j])
        img[i, j] = white
        dfs(i - 1, j, img[i, j], img, points)
        dfs(i + 1, j, img[i, j], img, points)
        dfs(i, j - 1, img[i, j], img, points)
        dfs(i, j + 1, img[i, j], img, points)
        return points

def deskew(sub, points):
    h, w = sub.shape[0], sub.shape[1]
    top = -1
    top_line = [-1] * w
    for i in xrange(0, h):
        for j in xrange(0, w):
            if sub[i, j] < threshold:
                if top == -1:
                    top = j
                if top_line[j] == -1:
                    top_line[j] = i
    if top < w * 0.4:
        for i in xrange(top + 1, w):
            if top_line[i] > top_line[i - 1]:
                break
        hori = float(w - 1 - i)
        vert = float(top_line[w - 1] - top_line[i])
        if hori > 10.0:
            theta = np.tanh(vert/hori) * 57.29
        else:
            theta = 0.0
    elif top > w * 0.6:
        for i in xrange(top, 1, -1):
            if top_line[i - 1] > top_line[i]:
                break
        hori = float(i)
        vert = float(top_line[i] - top_line[0])
        if hori > 10.0:
            theta = np.tanh(vert/hori) * 57.29
        else:
            theta = 0.0
    else:
        theta = 0.0
    rotation_M = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1)
    rotated_im = cv2.warpAffine(sub, rotation_M, (w,h), flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
    return rotated_im

def splitReCaptcha(im, directory):
    outdir = directory + '/output/'
    img = cv2.imread(im, 0)
    backup = img.copy()
    h, w = img.shape[0], img.shape[1]
    count = 0

    filename = im.replace(".jpeg", "")
    for i in xrange(0, h):
        for j in xrange(0, w):
            if img[i, j] < threshold:
                points = []
                dfs(i, j, img[i, j], img, points)
                if len(points) > num_threshold:
                    min_x, min_y = np.min(points, axis=0)
                    max_x, max_y = np.max(points, axis=0)
                    if max_x - min_x < small_threshold or max_y - min_y < small_threshold:
                        continue
                    sub = backup[min_x : max_x + 1, min_y : max_y + 1]
                    img[min_x : max_x + 1, min_y : max_y + 1] = white
                    if max_x - min_x > singel_char:
                        sub = deskew(sub, points)
                    cv2.imwrite(os.path.join(outdir, '{0}_split{1}.jpeg'.format(filename, count)), sub)
                    count += 1

if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    directory = os.getcwd()
    indir = directory + '/input/'
    os.chdir(indir)
    outdir = directory + '/output/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for filename in os.listdir(indir):
        if filename.endswith(".jpeg"):
            splitReCaptcha(filename, directory)
