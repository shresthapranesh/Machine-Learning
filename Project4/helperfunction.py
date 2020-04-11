from scipy.ndimage import label, uniform_filter, find_objects, generate_binary_structure
from scipy.ndimage import binary_opening, binary_closing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift

from skimage.filters import threshold_otsu
from skimage import io
from skimage.measure import regionprops
from skimage.color import label2rgb
import math
import os
import cv2 as cv


# Bradley's Adaptive Thresholding Algorithm
def Adaptive_Threshold(image):
    percentage = 99.0 / 100.0
    window_diam = 60669  # block size of 1/8th image dimension
    img = np.array(image).astype(float)
    means = uniform_filter(img, window_diam, mode='nearest')
    height, width = img.shape[:2]
    result = np.zeros((height, width), np.uint8)
    if np.mean(means) > 160:
        result[img > percentage * means+25] = 1
    else:
        result[img > percentage * means+45] = 1
    return result


def padding(image):
    ht, wd = image.shape
    height = 40
    width = 40
    if image.shape[0] > height or image.shape[1] > width:
        img = cv.resize(image, (height, width))
    else:
        # centered padding with zeros
        xx = (width - wd)//2
        yy = (height - ht)//2
        img = np.zeros((height, width))
        img[yy:yy+ht, xx:xx+wd] = image
    return img


j = 1

# Absolute Filepath
raw_filepath = 'E:/Texas Tech University/Spring 2020/ECE-4332/Data/NemaLife Images_Converted/'
processed_path = 'E:/Texas Tech University/Spring 2020/ECE-4332/Data/processed image/'

filename = [imagename for imagename in os.listdir(raw_filepath)]


for imagename in filename:
    image = io.imread(raw_filepath + imagename, as_gray=True)

    # for illumination (source: Dr. HS-S Helper function)
    M, N = image.shape
    U, V = np.meshgrid(np.linspace(1, N, N), np.linspace(1, M, M))

    D = np.sqrt((U-(N+1)/2)**2 + (V-(M+1)/2)**2)
    D0 = 2
    n = 2

    one = np.ones((M, N))
    H = 1/(one+(D/D0)**(2*n))
    G = fftshift(fft2(image))*H
    g = np.real(ifft2(ifftshift(G)))
    out = np.float_(image) - g
    img = 255/(np.max(out)-np.min(out))*(out-np.min(out))
    img = np.uint(img)

    th = Adaptive_Threshold(img)
    s = generate_binary_structure(2, 2)
    # remove small white spots
    test1 = binary_opening(th, structure=np.ones((2, 2)))

    # labels connected pixels
    label_image, size = label(test1)

    # image_label_overlay = label2rgb(label_image, image=image)

    #fig, ax = plt.subplots(figsize=(10, 6))
    #ax.imshow(image, cmap=plt.cm.gray)
    for region in regionprops(label_image):
        if region.area > 150 and region.area < 5000:
            minr, minc, maxr, maxc = region.bbox
            image_patch1 = image[minr:maxr, minc:maxc]
            image_patch2 = test1[minr:maxr, minc:maxc]
            #value = np.mean(image_patch1)
            value2 = np.mean(image_patch2)

            #print('{}: mean: {}'.format(j, value2))
            if value2 < 0.15 or value2 >= 0.40:
                img2 = padding(image_patch1)
                plt.imsave(processed_path+'/noworm/'+str(j) +
                           '.png', img2, cmap=plt.cm.gray)
            elif region.area < 200 or region.area > 3000:
                img2 = padding(image_patch1)
                plt.imsave(processed_path+'/noworm/'+str(j) +
                           '.png', img2, cmap=plt.cm.gray)
            else:
                img2 = padding(image_patch1)
                plt.imsave(processed_path+'/worm/'+str(j) +
                           '.png', img2, cmap=plt.cm.gray)
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                           fill=False, edgecolor='red', linewidth=1)
            # ax.add_patch(rect)
            j += 1

    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()
