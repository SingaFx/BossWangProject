import numpy as np
import cv2
import os
import sys

white = 255
black = 0
white_threshold = 238 # white background
diff = 230
num_threshold = 20 # noise
small_threshold = 8
singel_char = 50

SEARCH_WHITE = 0
SEARCH_NONWHITE = 1

def abs_diff(a, b):
    return (a - b) if a > b else (b - a)

def dfs(i, j, prev, img, points, threshold, flag=SEARCH_NONWHITE):
    h, w = img.shape[0], img.shape[1]
    if i >= 0 and j >= 0 and i < h and j < w and \
        (flag == SEARCH_NONWHITE) == (img[i, j] < threshold):
        points.append([i, j])
        img[i, j] = white if flag == SEARCH_NONWHITE else black
        dfs(i - 1, j, img[i, j], img, points, threshold, flag)
        dfs(i + 1, j, img[i, j], img, points, threshold, flag)
        dfs(i, j - 1, img[i, j], img, points, threshold, flag)
        dfs(i, j + 1, img[i, j], img, points, threshold, flag)
    return points

def deskew(sub, points):
    h, w = sub.shape[0], sub.shape[1]
    top = -1
    top_line = [-1] * w
    for i in xrange(0, h):
        for j in xrange(0, w):
            if sub[i, j] < white_threshold:
                if top == -1:
                    top = j
                if top_line[j] == -1:
                    top_line[j] = i
    if top < w * 0.48:
        for i in xrange(top + 1, w):
            if top_line[i] > top_line[i - 1]:
                break
        hori = float(w - 1 - i)
        vert = float(top_line[w - 1] - top_line[i])
        if hori > 10.0:
            theta = np.tanh(vert/hori) * 57.29
        else:
            theta = 0.0
    elif top > w * 0.52:
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

def remove_white_space(img):
    h, w = img.shape
    whites = []
    non_whites = []
    top, bottom, left, right = -1, h, -1, w
    # top
    for i in xrange(0, h / 2):
        num_whites = 0
        for j in xrange(0, w):
            if img[i, j] > white_threshold:
                num_whites += 1
        if num_whites > w / 3:
            top = i
        else:
            break
    # bottom
    for i in xrange(h - 1, h / 2 + 1, -1):
        num_whites = 0
        for j in xrange(0, w):
            if img[i, j] > white_threshold:
                num_whites += 1
        if num_whites > w / 3:
            bottom = i
        else:
            break
    offset = top + h - bottom
    # left
    for i in xrange(0, w / 2):
        num_whites = 0
        for j in xrange(0, h):
            if img[j, i] > white_threshold:
                num_whites += 1
        if num_whites > (h + offset) / 3:
            left = i
        else:
            break
    # right
    for i in xrange(w - 1, w / 2 + 1, -1):
        num_whites = 0
        for j in xrange(0, h):
            if img[j, i] > white_threshold:
                num_whites += 1
        if num_whites > (h + offset) / 3:
            right = i
        else:
            break
    return img[top + 1:bottom, left + 1:right]

def fill_white_space(img):
    h, w = img.shape
    samples = np.concatenate((img[0, :], img[h - 1, :]))
    darkest = np.median(samples)
    points = []
    th = darkest + (white_threshold - darkest) * 3 / 4
    dfs(0, 0, img[0, 0], img, points, th, SEARCH_WHITE)
    dfs(0, w - 1, img[0, w - 1], img, points, th, SEARCH_WHITE)
    dfs(h - 1, 0, img[h - 1, 0], img, points, th, SEARCH_WHITE)
    dfs(h - 1, w - 1, img[h - 1, w - 1], img, points, th, SEARCH_WHITE)
    for i, j in points:
        img[i, j] = darkest
    return img

def deskew_to_gray(im, outdir, white_out_dir):
    img = cv2.imread(im, 0)
    backup = img.copy()
    h, w = img.shape[0], img.shape[1]
    count = 0
    filename = im.replace(".jpeg", "")
    fills = []
    for i in xrange(0, h):
        for j in xrange(0, w):
            if img[i, j] < white_threshold:
                points = []
                dfs(i, j, img[i, j], img, points, white_threshold)
                if len(points) > num_threshold:
                    min_x, min_y = np.min(points, axis=0)
                    max_x, max_y = np.max(points, axis=0)
                    if max_x - min_x < small_threshold or max_y - min_y < small_threshold:
                        continue
                    sub = backup[min_x : max_x + 1, min_y : max_y + 1]
                    img[min_x : max_x + 1, min_y : max_y + 1] = white
                    cv2.imwrite(os.path.join(outdir, '{0}_split_{1}.jpeg'.format(filename, count)), sub)
                    if max_x - min_x > singel_char:
                        sub = deskew(sub, points)
                        cv2.imwrite(os.path.join(outdir, '{0}_deskew_{1}.jpeg'.format(filename, count)), sub)
                        sub = remove_white_space(sub)
                        cv2.imwrite(os.path.join(white_out_dir, '{0}_remove_white_{1}.jpeg'.format(filename, count)), sub)
                        sub = fill_white_space(sub)
                    fillname = os.path.join(white_out_dir, '{0}_fill_white_{1}.jpeg'.format(filename, count))
                    cv2.imwrite(fillname, sub)
                    fills.append(fillname)
                    count += 1
    # generally fill white space has very little effect, so do not show it for simplicity
    return fills

def preprocess_raw_image(path):
    sys.setrecursionlimit(20000)
    return deskew_to_gray(path, "output/", "output/")

if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    directory = os.getcwd()
    indir = directory + '/input/'
    os.chdir(indir)
    outdir = directory + '/output/'
    white_out_dir = directory + '/rm_white/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(white_out_dir):
        os.makedirs(white_out_dir)
    for filename in os.listdir(indir):
        if filename.endswith(".jpeg"):
            deskew_to_gray(filename, outdir, white_out_dir)
