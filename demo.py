# Copyright (C) 2013,2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import cv2
import numpy as np

from JguPy import *

# Input high resolution RGB image
cImg = cv2.imread("view5.png")
# Input low resolution depth image with a resolution that is
# 4x4 times smaller than the RGB image
dimg = cv2.imread("disp5_4ds.png", 0)
# Parameters:
# upsampleRate	: upsampleing rate
# sigma 		: the bandwidth parameter for the Gaussian kernel
# lambda1 		: the weighting factor on the color difference.
# interval 		: approximate optimization parameter.
upsampleRate = 4
interval = 3
sigma = 0.5
lambda1 = 10.0

jgu = jgu()
uImg = jgu.Upsample(np.uint8(cImg), np.uint8(dimg), np.int(upsampleRate), np.int(interval), sigma, lambda1)[0]

cv2.imshow("input RGB image", cImg)
cv2.imshow("input low resolution depth image", dimg)
cv2.imshow("output hight resolution depth image", uImg)
cv2.waitKey()
cv2.destroyAllWindows()
