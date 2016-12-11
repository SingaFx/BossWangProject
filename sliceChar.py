import numpy as np
import cv2
import os
np.set_printoptions(threshold='nan')

def line_array(array):
    list_x = []
    for y in range(len(array)):
        if all(i >= 3 for i in array[y:y+9]) == True:
            list_x.append(y-1)
    return list_x



def sliceChar(f, directory):
    filename = f.rsplit('.')[0]

    #-------------Thresholding Image--------------#

    src_img = cv2.imread(f, 1)
    # copy = src_img.copy()
    # src_img = cv2.resize(copy, dsize =(1500, 1000), interpolation = cv2.INTER_AREA)
    height = src_img.shape[0]
    width = src_img.shape[1]
    print("#----------------------------#")
    print("Image Info:-")
    print("Height =",height,"\nWidth =",width)
    print("#----------------------------#")

    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    gud_img = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 101, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    noise_remove = cv2.erode(gud_img,kernel,iterations = 2)

    kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)

    opening = cv2.morphologyEx(gud_img, cv2.MORPH_OPEN, kernel, iterations = 2) # To remove "pepper-noise"
    kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)
    final_thr = cv2.dilate(noise_remove, kernel1, iterations = 3)

    count_x = np.zeros(shape= (height))
    for y in range(height):
        for x in range(width):
            if noise_remove[y][x] == 255 :
                count_x[y] = count_x[y]+1

    line_list = line_array(count_x)


    #-------------Character segmenting------------#

    chr_img = final_thr.copy()

    contours, hierarchy = cv2.findContours(chr_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # final_contr = np.zeros((final_thr.shape[0],final_thr.shape[1],3), dtype = np.uint8)
    # cv2.drawContours(final_contr, contours, -1, (0,255,0), 3)

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 100:
            x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(src_img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imwrite(os.path.join(outdir, '{0}_letter_{1}.jpeg'.format(filename, i)), src_img[y:y + h, x:x + w])


if __name__ == '__main__':
    directory = os.getcwd()
    indir = directory + '/binary/'
    os.chdir(indir)
    outdir = directory + '/char/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for filename in os.listdir(indir):
        print filename
        if filename.endswith(".jpeg"):
            try:
                sliceChar(filename, directory)
            except Exception as e:
                print e
