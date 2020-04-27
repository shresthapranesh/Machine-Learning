import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['figure.dpi'] = 20

filenames = []
total_images = os.walk('./NemaLife Images_Converted/')
print("Loading Files...")
for (a, b, files) in total_images:
    for f in files:
        filenames.append(f)
    print(f"Files: {len(filenames)}")

# Contrast Stretch
mean_vect = []
print("Contrast Stretching...")
for f in filenames:
    img = cv.imread(f'./NemaLife Images_Converted/{f}', 0)
    mean_value = np.mean(img)
    mean_vect.append(mean_value)
mean_brightness = np.mean(mean_vect)
print(f"Absolute Mean: {mean_brightness}")

for f in filenames:
    img = cv.imread(f'./NemaLife Images_Converted/{f}', 0)
    actual_mean = np.mean(img)
    diff = abs(mean_brightness - actual_mean)
    if mean_brightness > actual_mean:
        img = abs(img + diff)
    else:
        img = abs(img - diff)
    cv.imwrite(f"./ContrastStretch/{f}", img)


def pad_images_to_same_size(images):
    h, w = images.shape[:2]
    width_max = max(40, w)
    height_max = max(40, h)
    if width_max > 40:
        dim = (40, round(h * 40 / w))
        images = cv.resize(images, dim, interpolation=cv.INTER_AREA)
    if height_max > 40:
        dim = (round(40 * w / h), 40)
        images = cv.resize(images, dim, interpolation=cv.INTER_AREA)

    h, w = images.shape[:2]
    width_max = max(40, w)
    height_max = max(40, h)

    diff_vert = height_max - h
    pad_top = diff_vert // 2
    pad_bottom = pad_top
    diff_hori = width_max - w
    pad_left = diff_hori // 2
    pad_right = pad_left
    img_padded = cv.copyMakeBorder(
        images, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT, value=0)

    check_h, check_w = img_padded.shape

    if (check_h <= 40 and check_w <= 40):
        if check_h != 40:
            img_padded = cv.copyMakeBorder(
                img_padded, 1, 0, 0, 0, cv.BORDER_CONSTANT, value=0)
        if check_w != 40:
            img_padded = cv.copyMakeBorder(
                img_padded, 0, 0, 1, 0, cv.BORDER_CONSTANT, value=0)
        return img_padded
    else:
        return pad_images_to_same_size(img_padded)


# MAIN LOOP
ite = 0
worm_og = []
no_worm_og = []
num = 0
path_noworm = "./Output/NoWorm/"
path_worm = "./Output/Worm/"
print("Generating Dataset...")
for images in filenames:
    num += 1
    img = cv.imread(f'./ContrastStretch/{images}', 0)
    img_og = img  # original copy for cropping contours

    # adptiveThreshold of Contrast Stretched images
    th1 = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 99, -35, None)

    contours, hierarchy = cv.findContours(
        th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cropping the contours
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if w * h < 300 or w * h > 4900:  # disregards boxes
            continue

        cropped = img_og[y:y + h, x:x + w]
        cropped_check = th1[y:y + h, x:x + w]
        # check which class the boxed image go to
        # pad/resize the cropped image
        # save to respective directory
        if np.mean(cropped_check) < 85 and np.mean(cropped_check) > 40:
            cropped = pad_images_to_same_size(cropped)
            cv.imwrite(f'{path_worm}{ite}.jpg', cropped)
            _, cropped = cv.threshold(cropped, 200, 255, cv.THRESH_BINARY)
            worm_og.append(np.mean(cropped))
        elif np.mean(cropped_check) < 35 or np.mean(cropped_check) > 95:
            cropped = pad_images_to_same_size(cropped)
            cv.imwrite(f'{path_noworm}{ite}.jpg', cropped)
            _, cropped = cv.threshold(cropped, 200, 255, cv.THRESH_BINARY)
            no_worm_og.append(np.mean(cropped))
        ite += 1

    # loop to show boxed images
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if w * h < 400 or w * h > 4900:
            continue
        boxed_img = cv.rectangle(
            img_og, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.imshow("img", boxed_img); cv.waitKey(0); cv.destroyAllWindows()
