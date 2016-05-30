#coding= utf-8
__author__ = 'yujinke'
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import os.path
import copy









def swap(hw,angle):
        if(angle < -45 ):
            hw[0],hw[1] = hw [1],hw [0];
        else:
            hw[0],hw[1]  = hw [0],hw [1];





def compound_new(p1,hw,angle):

     A = [-1,-1];

     if(angle < -45 ):
        A[0],A[1] = hw[1],hw[0];
     else:
        A[0],A[1] = hw[0],hw[1];


     p1 = (int(p1[0]),int(p1[1]));
     p2 = (int(p1[0]+A[0]),int(p1[1]+A[1]));

     return (p1,p2)



def ver_h_w_ratio(hw,angle):

    A = [-1,-1];

    if hw[0]<10 or hw[1]<10:
        return False;

    if(angle < -45 ):
        A[0],A[1] = hw [1],hw [0];
    else:
        A[0],A[1]  = hw [0],hw [1];

    ratio = A[1]/(A[0]+0.01);

    if(ratio<6 and ratio>1.7 and A[0]>5):

        return A

    return False





def verify(P):

    """ verify the segment img is a char by machine learning [ SVM ]
     cnn call from """
    centers = [];
    widths = [];
    heights = [];

    for x in P:
        rotRect =  cv2.minAreaRect(x);

        A =  [-1,-1];

        if(ver_h_w_ratio(rotRect[1],rotRect[2],A) ):

            center = [rotRect[0][0]+rotRect[1][0]/2,rotRect[0][1]+rotRect[1][1]/2];
            centers.append(center[1]);
            widths.append(A[0]);
            heights.append(A[1]);
        else:
            return -1;

















def select(A):
    lena = len(A)
    data = []
    for a in range(0,lena,1):
        for b in range(a+1,lena,1):
            for c in range(b+1,lena,1):
                data.append([A[a],A[b],A[c]])
    return data;


# def check(P):
#     choice = select(P);
#     for x in choice:
#         for p in choice:


def deskew(img,SZ):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=cv2.affine_flags)
    return 0

def subthres(img1,cube_x,cube_y):
            img2 = np.zeros(img1.shape,dtype=np.uint8);
            img1 = cv2.dilate(img1,None)
            for i in range(0,192,cube_x):
                for j in range(0,36,cube_y):

                    new = img1[j:j+cube_y,i:i+cube_x]
                    m,t = cv2.threshold(new,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    img2[j:j+cube_y,i:i+cube_x] =t




            return img2





def CM(S):
    a = np.arange(1,S+1,1)
    return a[S%a == 0]


def estimate(P):
    A = [-1,-1,-1];

    A[0] = P[0][0][0];
    A[1] = P[1][0][0];
    A[2] = P[2][0][0];
    A.sort();
    diff_A = A[1] - A[0];
    diff_B = A[2] - A[1];
    error  = abs(diff_B - diff_A)*abs( 96 - A[0]);
    print(error);

    return error;




def r2p1(r):
    p1 = (r[0],r[1]);
    return p1

def r2p2(r):
    p2 = (r[0]+r[2],r[1]+r[3]);
    return p2

def computediff(a,b):
    list =  [];

    for i in range(1,len(a)):
        diff = abs(b[i-1][0]-a[i][0]);
        list.append(diff);
    return list;


def CompRect(rect1,rect2):
    p2 = (rect1[0]+rect1[2],rect1[1]+rect1[3]);
    p1 = (rect2[0],rect2[1])

    print "comp r1,r2",rect1,rect2;

    dst_saqure = rect2[2]*rect2[3]
    src_saqure = float((p2[1]-p1[1])*(p2[0]-p1[0]));
    print "rate:",src_saqure/dst_saqure;

    return src_saqure/dst_saqure;





dir_folder = "./plate_repick"
for parent,dirnames,filenames in os.walk(dir_folder):
    for filename in filenames:
        if filename.endswith(".jpg"):
            filepath = parent+"/"+filename
            print filepath
            img1 = cv2.imread(filepath,cv2.CV_LOAD_IMAGE_GRAYSCALE);
            img1_color = cv2.imread(filepath);
            blur = cv2.GaussianBlur(img1,(5,5),0)
            x  = [12,16,24,32]
            y  = [6,12,18,36]
            print x,y
            new = subthres(img1,16,18);
            print img1.dtype,new.dtype;
            new.dtype = img1.dtype;



            ret,new = cv2.threshold(new,70,255,cv2.THRESH_BINARY);


            val  =  new.sum(axis = 0);
            vel = new.sum(axis = 1);
            thes_c = new.__deepcopy__(new);


            contours = cv2.findContours(thes_c,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);

            list = contours[0];
            scontainer = [];
            scontainer_bounding = [];

            for one in list:

                rot_rect = cv2.minAreaRect(one);
                rot_rect_bounding = cv2.boundingRect(one);

                if(ver_h_w_ratio(rot_rect[1],rot_rect[2]) ):
                    scontainer.append((rot_rect[0],ver_h_w_ratio(rot_rect[1],rot_rect[2])));
                    scontainer_bounding.append(rot_rect_bounding);



            scontainer.sort(key = lambda x:[x][0][0])
            print "sconationer_bounding",scontainer_bounding
            scontainer_bounding.sort(key = lambda  x:[x][0]);
            min_ = np.array([ abs(x[0]-96) for x in scontainer_bounding ]);
            mid = min_.argmin()



            for m in range(mid,0,-1):
                i_scontainer_bounding = scontainer_bounding[m];
                i_scontainer_bounding_l = scontainer_bounding[m-1];
                diffx = i_scontainer_bounding[0] - (i_scontainer_bounding_l[0]+i_scontainer_bounding_l[2])
                arg_ = (i_scontainer_bounding[2] + i_scontainer_bounding[2])/2
                dif = scontainer_bounding[m+1][0] - (scontainer_bounding[m][0]+scontainer_bounding[m][2])+4

                if(diffx % arg_ > dif  and abs(i_scontainer_bounding_l[0]-49)<20):
                    result = i_scontainer_bounding_l;
                    break;

            if(result == []):
                min_ = np.array([ abs(x[0]-49) for x in scontainer_bounding ]);
                mid_g = min_.argmin()
                result = scontainer_bounding[mid_g]
            print "result",result




            sc_p1 = [r2p1(x) for x in scontainer_bounding];
            sc_p2 = [r2p2(x) for x in scontainer_bounding];
            for i in range(len(scontainer_bounding)):
                cv2.rectangle(img1,sc_p1[i],sc_p2[i],(255));


            #按照 x 序列排序.


            x = range(len(val));
            y = range(len(vel));
            plt.subplot('221');
            plt.plot(x,val);
            plt.subplot('223')
            plt.plot(y,vel);
            plt.subplot('222');
            plt.imshow(new,cmap='gray');
            plt.subplot('224')
            plt.imshow(img1);




#x 16 # y 18 will be good


#             for m in range(len(x)):
#                 for n in range(len(y)):
#                     plt.subplot(len(x),len(y),m*len(x)+ n);
#                     A = subthres(img1,x[m],y[n])
#                     plt.axis('off')
#                     plt.title("x:"+str(x[m])+",y:"+str(y[n]))
#                     plt.imshow(A,cmap='gray');







            plt.show();










