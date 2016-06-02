# coding=utf-8
__author__ = 'yujinke'
import cv2;
import os;
import os.path;
import hashlib
import numpy as np;



def r(factor):
    return int(np.random.random()*factor);
def addRandomLines(src):
    l = min(src.shape);


    cv2.line(src,(r(30),0),(128-r(30),0),(255,255,255),r(15));
    cv2.line(src,(r(30),128),(128-r(30),128),(255,255,255),r(15));
    for i in range(r(10)):
        color = r(255)
        p1 = (r(l),r(l))
        p2 = (r(l),r(l))
        cv2.line(src,p1,p2,(color,color,color),2);
    return src;



def SaltAndPepper(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.random_integers(0,src.shape[0]-1)
        randY=np.random.random_integers(0,src.shape[1]-1)
        if np.random.random_integers(0,1)==0:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255


    return NoiseImg

def GuessNoise(scr,val):
    param=val
    #灰阶范围
    grayscale=256
    w=scr.shape[1]
    h=scr.shape[0]
    newimg=np.zeros((h,w),np.uint8)


    for x in xrange(0,h):
        for y in xrange(0,w,2):
            r1=np.random.random_sample()
            r2=np.random.random_sample()
            z1=param*np.cos(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
            z2=param*np.sin(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))

            fxy=int(scr[x,y]+z1)
            fxy1=int(scr[x,y+1]+z2)
            #f(x,y)
            if fxy<0:
                fxy_val=0
            elif fxy>grayscale-1:
                fxy_val=grayscale-1
            else:
                fxy_val=fxy
            #f(x,y+1)
            if fxy1<0:
                fxy1_val=0
            elif fxy1>grayscale-1:
                fxy1_val=grayscale-1
            else:
                fxy1_val=fxy1
            newimg[x,y]=fxy_val
            newimg[x,y+1]=fxy1_val
    return newimg;


def imgComb(body,A,B,dis,rangex):
    A = cv2.resize(A,(64,128));
    B = cv2.resize(B,(64,128))
    body = cv2.resize(body,(64,128));
    A = A[0:128,dis:64-dis]
    B = B[0:128,dis:64-dis]
    body = body[0:128,dis:64-dis]
    generate = np.hstack((A,body,B))
    center = generate.shape[1]/2
    length = int(rangex * np.random.random())
    left = center - ( 24 + length);
    right = center + (24 + length);
    result = generate[0:128,left:right];
    result = cv2.resize(result,(128,128));

    result = addRandomLines(result)
    result = SaltAndPepper(result,0.2)
    result = GuessNoise(result,70)
    return result;



def rotRandrom(img,factor,size):
    """ 使图像轻微的畸变

        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸

    """
    shape = size;
    pts1 = np.float32([[0,0],[0,shape[0]],[shape[1],0],[shape[1],shape[0]]])
    pts2 = np.float32([[r(factor),r(factor)],[0,shape[0]-r(factor)],[shape[1]-r(factor),0],[shape[1]-r(factor),shape[0]-r(factor)]])
    M  = cv2.getPerspectiveTransform(pts1,pts2);
    dst = cv2.warpPerspective(img,M,size);
    return dst;

    #return dst.ravel();




chaos_path_list = [];

classify_list = [];

def r(l):
    return int(np.random.random()*l);

def test(dir):
    x = 1;
    for parent,dirnames,filenames in os.walk(dir):
        images_list = [];
        if(parent.startswith(dir+"/")):
            for filename in filenames:
                path = parent + "/" + filename;
                if(path.endswith(".jpg") or path.endswith(".png")):
                    chaos_path_list.append(path)
                    images_list.append(path);
        classify_list.append(images_list);





test("./Fnt")

def md5(src):

    myMd5 = hashlib.md5()
    myMd5.update(src)
    myMd5_Digest = myMd5.hexdigest()
    return myMd5_Digest

def generateOne(chaospaths,Normalfolders):
    for Folder in Normalfolders:
        for img_path in Folder:
            lp = len(chaos_path_list)
            img1 = cv2.imread(img_path,cv2.CV_LOAD_IMAGE_GRAYSCALE);
            A = cv2.imread(chaospaths[r(lp)],cv2.CV_LOAD_IMAGE_GRAYSCALE);
            B = cv2.imread(chaospaths[r(lp)],cv2.CV_LOAD_IMAGE_GRAYSCALE);

            img1 = cv2.bitwise_not(img1)
            A = cv2.bitwise_not(A)
            B = cv2.bitwise_not(B)
            img1 = rotRandrom(img1,30,(128,128));
            A = rotRandrom(A,20,(128,128));
            B = rotRandrom(B,20,(128,128));
            result = imgComb(img1,A,B,r(20),r(20));
            folder,filename = os.path.split(img_path);
            print filename
            cv2.imwrite(folder+"/" + md5(filename)+".png",result);

generateOne(chaos_path_list,classify_list);














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
