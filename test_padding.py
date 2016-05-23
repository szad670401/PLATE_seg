#coding=utf-8
import numpy as np;

import cv2;

from matplotlib import pyplot as plt

import os;
import math;
import svm

dir_T = "./Char_classify/T";
dir_F = "./Char_classify/F";

dir_folder = "./HasPlate"

template = np.zeros((36,36,1),np.uint8);

def dis(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2);

pit = "";

factor = 0 ;

for r in xrange(36):
    for c in xrange(36):

        x = 1/math.exp(0.13*dis([r,c],[18,18])) ;

        template[r,c] = int( x*255 )


template = cv2.resize(template,(20,36))
cv2.imshow("demo",template);

template = template.astype(np.float64)/255;

model = svm.SVM(C=1.67, gamma=0.0383)
model.load('digits_svm.dat')

print "loaded"



def derivative(vector):
    A =  [];

    for i in xrange(1,len(vector)):
        A.append(vector[i]-vector[i-1])
    return A;

def subthres(img1,cube_x,cube_y):
            img2 = np.zeros(img1.shape,dtype=np.uint8);
            img1 = cv2.dilate(img1,None)
            for i in range(0,192,cube_x):
                for j in range(0,36,cube_y):

                    new = img1[j:j+cube_y,i:i+cube_x]
                    m,t = cv2.threshold(new,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    img2[j:j+cube_y,i:i+cube_x] =t




            return img2


def findpeak(vector,depth):
    """先找到第一个较低点  然后再找到第二个较低点 如果第二个点 比 第一个点低 且 投影直方图中 第二个点也比第一个点低 即选用第一个点   """
    switch_A =  0 ;
    head  = -1 ;
    tail  = -1 ;


    switch_B =  0 ;
    switch_C =  0 ;

    Points =  [] ;

    for a in xrange(1,len(vector)):
        if(vector[a-1]>vector[a]):
            #print a,vector[a],"↓"
            if( switch_A == 1 and switch_B == 1) :
                if(abs(vector[tail]  - vector[a]) > depth and a-head >15 ):
                    #找到点

                    #print "Point",tail,vector[tail],a - head,1/(vector[tail]*100)*(a - head)


                    Points.append([tail+10,vector[tail],a - head,1/(vector[tail]*100)*(a - head)])

                    switch_A = 0 ;
                    switch_B = 0 ;
            if(switch_A ==  0 ):
                head = a ;
                switch_A  = 1;

        if(vector[a]>vector[a-1] and switch_A == 1):
            #print a,vector[a],"↑"

            if(switch_B  == 0):
                tail = a - 1;

                if(abs( vector[head] - vector[tail]) > depth):
                    #print "switchB  =  1"
                    switch_B = 1;

#    Points = np.array(Points);
    #print Points[0][1],Points[1][1]
    #print abs(Points[0][1]-Points[1][1])


    if abs(Points[0][1]-Points[1][1]) < 0.003 or Points[0][1]>Points[1][1] :
        #print "Result",Points[1];
        result  = Points[1];

    else:

        result = Points[0]
    return result[0];

def findSmallPeak(points,fixed,step = 10,epoch = 20 ):
    points = np.array(points)
  #  min  =  points.argmin()
    c_pos = fixed;

    for i in range(epoch):


        if((c_pos - step) >= 0 and c_pos + step < len(points)):
            prev = points[c_pos - step];
            back = points[c_pos + step];
            if(prev <= back):
                symbol = -1;
            if(prev > back):
                symbol = 1;

            if(step > 1):
                step -=1


            #print "c_pos",c_pos,"symbol",symbol;
            c_pos+= symbol;

    return c_pos;

def verify(hw,angle):

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

def computeHistMax(A,min,max):
    range_x = np.zeros(max-min,dtype=np.int8)
    offset = min ;
    print A
    for one in A:
        if(min<one  and one<max):
            range_x[one-offset] += 1;
    print "range_X",range_x;

    print "A.argmax",range_x.argmax()+min;

    return range_x.argmax()+min




def char_seg(IMG,table):
    List =[];
    for i in xrange(len(table)):
        if(table[i] < 0 ):
            table[i] = 0 ;


    print table
    for i in xrange(1,len(table)):

        List.append(IMG[0:36,table[i-1]:table[i]])
    return List;

def plateSegment(plate):
    filter_hist = [];
    v_hist = [] ;


    for x in xrange(0,plate.shape[1] - 20,1):
        char = img1[0:36,x:x+20];
        we_char = char*template/(255*20*36)
        filter_hist.append(we_char.sum());

    sep = findpeak(filter_hist,0.003);
    print "sep",sep;

    v_project = plate.sum(axis =  0 );
    plate_thes  = subthres(plate,16,18)
    contours = cv2.findContours(plate_thes,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
    list = contours[0];
    width_list = [];

    for one in list:
        rot_rect = cv2.minAreaRect(one);
        rot_rect_bounding = cv2.boundingRect(one);
        width_list.append(rot_rect_bounding[2]);


    val_most = computeHistMax(width_list,10,21);

    char_gap = val_most*0.288; #字符间隙

    char_whole = char_gap + val_most ; #一个整体字符的宽度  A unit of Char with the gap between two chars and the width of chars
    next_point =  sep + char_whole;
    region_point = sep  - char_whole;
    region_point = findSmallPeak(v_project,region_point,1,10);
    province_Point = region_point - char_whole*0.95
    seq_points = [];
    seq_points.append(int(province_Point));
    seq_points.append(int(region_point));
    seq_points.append(sep);
    predict_seq_points =  []
    for x in range(5):
        predict_seq_points.append(next_point+char_whole*x)
    print "predict_seq_points",predict_seq_points
    for x in range(4):
        R  = 3;
        peak_small= findSmallPeak( v_project,int(next_point));
        error = abs(peak_small - next_point);

        optimized_point = (next_point + peak_small)/2;
        print "predict",next_point,"samll_peak",peak_small,"error:",error,"optimized:",optimized_point;

        next_point =  peak_small + char_whole
        seq_points.append(peak_small);

    seq_points.append(seq_points[len(seq_points)-1] + val_most);
    print "seq_points",seq_points

    char_imgs = char_seg(plate,seq_points);

    return char_imgs;






        #if(verify(rot_rect[1],rot_rect)):

for parent,dirnames,filenames in os.walk(dir_folder):
    for filename in filenames:
        if filename.endswith(".jpg"):
            filepath = parent+"/"+filename;
            img1 = cv2.imread(filepath,cv2.CV_LOAD_IMAGE_GRAYSCALE);
            counts = [];
           # print img1.shape;

            for x in xrange(0,img1.shape[1] - 20,1):
                char = img1[0:36,x:x+20];
                we_char = char*template/(255*20*36)
                counts.append(we_char.sum());

            counts2 = img1.sum(axis =  0 );
            counts_derv = abs(np.array(derivative(counts)))/(2*len(counts2));
            char_sets = plateSegment(img1);






            sum_derv  = counts_derv.sum();


            print "counts_derv",sum_derv;


            findpeak(counts,0.002)
            x1 =  range(len(counts));
            x2 =  range(len(counts2)) ;
            x3 =  range(len(counts_derv))

            plt.subplot( "412")
            plt.plot(x3,counts_derv);
            plt.subplot("413")
            plt.plot(x2,counts2) ;
            plt.subplot("414")
            plt.imshow(img1,cmap = "gray")
            plt.subplot("671")
            plt.imshow(char_sets[0]);
            plt.subplot("672")
            plt.imshow(char_sets[1]);
            plt.subplot("673")
            plt.imshow(char_sets[2]);
            plt.subplot("674")
            plt.imshow(char_sets[3]);
            plt.subplot("675")
            plt.imshow(char_sets[4]);
            plt.subplot("676")
            plt.imshow(char_sets[5]);
            plt.subplot("677")
            plt.imshow(char_sets[6]);

            plt.show()







