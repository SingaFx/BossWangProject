import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import glob

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids, file):
    bar = np.zeros((10, 100, 3), dtype = "uint8")
    startX = 0
    
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 100)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 10),
            np.repeat(color.astype("uint8"), 3).tolist(), -1)
        startX = endX    
    plt.imshow(bar)
    plt.savefig(file)

in_dir = "rm_white/"

for file in glob.glob(in_dir + "*fill_white*.jpeg"):
    img = cv2.imread(file)
    name = file[file.find('/') + 1: file.rfind('.')]
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey_img_reshape = grey_img.reshape((grey_img.shape[0] * grey_img.shape[1], 1))

    clt = KMeans(n_clusters = 2)
    clt.fit(grey_img_reshape)

    silhouette = metrics.silhouette_score(grey_img_reshape, clt.labels_, metric='euclidean')

    hist = centroid_histogram(clt)
    plot_colors(hist, clt.cluster_centers_, "kmeans/{}.jpeg".format(name))
    # ensure the first one is darker
    if clt.cluster_centers_[0] > clt.cluster_centers_[1]:
    	avg = clt.cluster_centers_[1] + (clt.cluster_centers_[0] - clt.cluster_centers_[1]) * 3 / 4
    	hist[0], hist[1] = hist[1], hist[0]
    else:
    	avg = clt.cluster_centers_[0] + (clt.cluster_centers_[1] - clt.cluster_centers_[0]) * 3 / 4
    print name, clt.cluster_centers_[0], clt.cluster_centers_[1], avg
    # assume two colors, and characters take up less space
    if hist[0] < hist[1]:
        new_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,39, 2)
        # _, new_img = cv2.threshold(grey_img, avg, 255, cv2.THRESH_BINARY)
    else:
    	new_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,39, 2)
    	# _, new_img = cv2.threshold(grey_img, avg, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(new_img, 'gray')
    plt.savefig("binary/{}.jpeg".format(name))


