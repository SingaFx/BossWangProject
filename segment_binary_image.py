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

num_black_lines = 4

min_w, min_h = 3, 10

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

def check_size(img):
    h, w = img.shape
    return True if w > min_w and h > min_h else False

def remove_all_black(img):
    h, w = img.shape
    whites = []
    non_whites = []
    top, bottom, left, right = -1, h, -1, w
    frac = 0.8
    # top
    for i in xrange(0, h / 2):
        num_whites = 0
        for j in xrange(0, w):
            if img[i, j] < white_threshold:
                num_whites += 1
        if num_whites > w * frac:
            top = i
        else:
            break
    # bottom
    for i in xrange(h - 1, h / 2 + 1, -1):
        num_whites = 0
        for j in xrange(0, w):
            if img[i, j] < white_threshold:
                num_whites += 1
        if num_whites > w * frac:
            bottom = i
        else:
            break
    offset = top + h - bottom
    # left
    for i in xrange(0, w / 2):
        num_whites = 0
        for j in xrange(0, h):
            if img[j, i] < white_threshold:
                num_whites += 1
        if num_whites > h * frac:
            left = i
        else:
            break
    # right
    for i in xrange(w - 1, w / 2 + 1, -1):
        num_whites = 0
        for j in xrange(0, h):
            if img[j, i] < white_threshold:
                num_whites += 1
        if num_whites > h * frac:
            right = i
        else:
            break
    succeed = (top < num_black_lines and left < num_black_lines and (w - right) < num_black_lines and (h - bottom) < num_black_lines)
    return img[top + 1:bottom, left + 1:right], succeed

def splitReCaptcha(im, directory, outdir, white_out_dir):
    raw_img = cv2.imread(im, 0)
    
    _, img = cv2.threshold(raw_img,127,255,cv2.THRESH_BINARY)
    img, succeed = remove_all_black(img)
    if not succeed:
        _, img = cv2.threshold(raw_img,127,255,cv2.THRESH_BINARY_INV)
        img, succeed = remove_all_black(img)
    # cannot segment
    if not succeed:
        return

    backup = img.copy()
    h, w = img.shape[0], img.shape[1]
    count = 0
    filename = im.replace(".jpeg", "")
    if h > w + 1:
       cv2.imwrite(os.path.join(outdir, '{0}_final.jpeg'.format(filename)), img)
       return
    for i in xrange(0, h):
        for j in xrange(0, w):
            if img[i, j] < white_threshold:
                points = []
                dfs(i, j, img[i, j], img, points, white_threshold, SEARCH_NONWHITE)
                if len(points) > num_threshold:
                    min_x, min_y = np.min(points, axis=0)
                    max_x, max_y = np.max(points, axis=0)
                    if max_x - min_x < small_threshold or max_y - min_y < small_threshold:
                        continue
                    sub = backup[min_x : max_x + 1, min_y : max_y + 1]
                    img[min_x : max_x + 1, min_y : max_y + 1] = white
                    cv2.imwrite(os.path.join(outdir, '{0}_final_{1}.jpeg'.format(filename, count)), sub)
                    count += 1

if __name__ == '__main__':
    sys.setrecursionlimit(20000)
    directory = os.getcwd()
    indir = directory + '/binary/'
    os.chdir(indir)
    outdir = directory + '/binary_out/'
    white_out_dir = directory + '/binary_white/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(white_out_dir):
        os.makedirs(white_out_dir)
    for filename in os.listdir(indir):
        if filename.endswith(".jpeg"):
            splitReCaptcha(filename, directory, outdir, white_out_dir)
