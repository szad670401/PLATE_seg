__author__ = 'yujinke'
import numpy as np


import cv2;
SZ = 180
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    print skew
    M = np.float32([[1, -5*skew, 5*SZ*skew], [0, 1, 0]])
    print M
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    cv2.imshow("xx",img);
    return img

x =  cv2.imread("test4.png",cv2.CV_LOAD_IMAGE_GRAYSCALE);
x = cv2.resize(x,(180,180))
cv2.imshow("y",x);
p = deskew(x);
cv2.imshow("x",p);
cv2.waitKey(0)
