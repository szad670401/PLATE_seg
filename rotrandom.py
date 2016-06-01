# coding=utf-8
__author__ = 'yujinke'
import cv2;
import numpy as np;



def r(factor):
    return np.random.random()*factor;

def rotRandrom(img,factor,size):
    """ 使图像轻微的畸变

        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸

    """
    img = img.reshape(size);
    shape = size;

    pts1 = np.float32([[0,0],[0,shape[0]],[shape[1],0],[shape[1],shape[0]]])
    pts2 = np.float32([[r(factor),r(factor)],[0,shape[0]-r(factor)],[shape[1]-r(factor),0],[shape[1]-r(factor),shape[0]-r(factor)]])
    M  = cv2.getPerspectiveTransform(pts1,pts2);
    dst = cv2.warpPerspective(img,M,(shape[0],shape[1]));

    return dst.ravel();



img1 = cv2.imread("./Fnt/Sample011/img011-00001.png",cv2.CV_LOAD_IMAGE_GRAYSCALE);
cv2.imshow("rot",rotRandrom(img1,1,[50,50]),);
cv2.waitKey(0);



# def rotRandrom(img,factor,size):
#     """ 使图像轻微的畸变
#
#         img 输入图像
#         factor 畸变的参数
#         size 为图片的目标尺寸
#
#     """
#     img = cv2.resize(img,size);
#
#     return img.ravel();
#
#
