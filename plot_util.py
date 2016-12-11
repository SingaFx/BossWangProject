from matplotlib import pyplot as plt
import numpy as np

def plot_images(paths):
    images = [cv2.imread(path) for path in paths]
    # assume num >= 2
    num = len(images)
    size = int(np.ceil(np.sqrt(num)))
    if num == 2:
        for i in xrange(num):
            plt.subplot(1,2,i+1), plt.imshow(images[i])
            plt.xticks([]),plt.yticks([])
    elif num == 3:
        for i in xrange(num):
            plt.subplot(1,3,i+1), plt.imshow(images[i])
            plt.xticks([]),plt.yticks([])
    else:
        for i in xrange(num):
            plt.subplot(size, size, i+1), plt.imshow(images[i])
            plt.xticks([]),plt.yticks([])
    plt.show()