__author__ = 'yujinke'
import numpy as np
import cv2
import os
import cPickle
import hashlib

def cutpadding(img,val):
    shape = img.shape;




    return img[0:shape[0],val:shape[1]-val];

    # 36 * 36

def getF(img):
    shape = img.shape;
    factor_random = np.random.random()* 6 - 3 ;

    F = img[0:shape[0],11+factor_random:29+factor_random];
    return cv2.resize(F,(18,36));
def md5(src):

    myMd5 = hashlib.md5()
    myMd5.update(src)
    myMd5_Digest = myMd5.hexdigest()
    return myMd5_Digest

def merge_(dirn ):
    set = [ ]

    def findinside(dirname):
        print dirname
        for parent,dirnames,filenames in os.walk(dirname):
            for filename,i in zip(filenames,range(len(filenames))):
                path = parent + "/" + filename ;
                if(path.endswith(".jpg") or path.endswith(".png")):
                        img = cv2.imread(path,cv2.CV_LOAD_IMAGE_GRAYSCALE);
                        #img = img.astype(np.float32)/255 ;
                        img = cv2.resize(img,(20,36));
                        if(dirname.find("zh_")==-1):
                            img = cv2.bitwise_not(img);

                        cv2.imwrite("./Char_classify/T/"+md5(img)+".jpg",img);

                if(i>1000):
                    break;

    for parent,dirnames,filenames in os.walk(dirn):
        for dirname in dirnames:
            c_path = dirn + "/" + dirname;
            findinside(c_path);







def merge(dirn ):
    set = [ ]
    print dirn.find("zh_");

    def findinside(dirname):
        print dirname
        for parent,dirnames,filenames in os.walk(dirname):
            for filename in filenames:
                path = parent + "/" + filename ;
                if(path.endswith(".jpg") or path.endswith(".png")):
                        img = cv2.imread(path,cv2.CV_LOAD_IMAGE_GRAYSCALE);
                        img = cv2.resize(img,(50,50));
                        #img = img.astype(np.float32)/255 ;
                        img = cutpadding(img,5);
                        img = cv2.resize(img,(20,36));

                        if(dirname.find("zh_") >-1):
                            set.append([img,1]);
                        else:
                            set.append([img,0]);

    for parent,dirnames,filenames in os.walk(dirn):
        for dirname in dirnames:
            c_path = dirn + "/" + dirname;
            findinside(c_path);
        count = 0 ;

    while(1):
        count+=1;


        L  = set[int(np.random.random() * len(set))]
        R  = set[int(np.random.random() * len(set))]
        print L[1],R[1];


        if(L[1] == 1 and R[1] == 0):
            R[0] = cv2.bitwise_not(R[0]);
            a = np.hstack([L[0],R[0]])
            cv2.imshow("a",a);
            F = getF(a);
            cv2.imshow("F",F);

        if(L[1] == 0 and R[1] == 0):

            a = np.hstack([L[0],R[0]])
            a = cv2.bitwise_not(a);
            cv2.imshow("a",a);
            F = getF(a);
            cv2.imshow("F",F);

        cv2.imwrite("./Char_classify/F/"+str(count)+".jpg",F);

        cv2.waitKey(0)




merge_("./chars2")
