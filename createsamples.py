__author__ = 'yujinke'
import numpy as np;

import cv2;

import matplotlib as plt

import svm
import os;
import cPickle
dir_T = "./Char_classify/T";
dir_F = "./Char_classify/F";

dir_folder = "./HasPlate"
model = svm.SVM(C=1.67, gamma=0.0383)
model.load('digits_svm.dat')


# f = open("./data.pkl", 'rb')
# training_data, validation_data, test_data = cPickle.load(f)
# samples_train, labels_train= test_data
#
# for i in range(100):
#
#      print "result",model.predict_single(samples_train[i]),labels_train[i]
#      samples_train[i]=  samples_train[i]*255
#
#      P  = samples_train[i].astype(np.uint8)
#
#      P =P.reshape((28,28))
#
#      cv2.imshow("smaples_train",P)
#
#      cv2.waitKey(0);


#test
A = cv2.resize(cv2.imread("./Char_classify/F/48.jpg",cv2.CV_LOAD_IMAGE_GRAYSCALE),(28,28)).ravel().astype(np.float32)
A = cv2.bitwise_not(A)

A = A/255

print ""
print "test",model.predict_single(A)

def subthres(img1,cube_x,cube_y):
            img2 = np.zeros(img1.shape,dtype=np.uint8);
            img1 = cv2.dilate(img1,None)
            for i in range(0,192,cube_x):
                for j in range(0,36,cube_y):

                    new = img1[j:j+cube_y,i:i+cube_x]
                    m,t = cv2.threshold(new,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    img2[j:j+cube_y,i:i+cube_x] =t




            return img2




def findcontiousNum(A):
    b = False;
    count = 0 ;

    for i in  A:
        if i == 1 and b == False :
            b = True;
            count+=1;
        if i == 0 and b == True :
            b = False;
    print count;

    return count;







for parent,dirnames,filenames in os.walk(dir_folder):
    for filename in filenames:
        if filename.endswith(".jpg"):


            filepath = parent+"/"+filename;
            img1 = cv2.imread(filepath,cv2.CV_LOAD_IMAGE_GRAYSCALE);
            #img1 = subthres(img1,16,18)
            #img1 = cv2.equalizeHist(img1)

            cv2.imshow("img1",img1);

            seq = [];

            for x in xrange(0,img1.shape[1] - 20,1):
                char = img1[0:36,x:x+15];
                char = cv2.resize(char,(28,28));
                char = cv2.bitwise_not(char)
                cv2.imshow("char",char);

                char = char.ravel().astype(np.float32)/255


                seq.append(model.predict_single(char)[0]);

                if model.predict_single(char)[0] == 1 :
                    cv2.waitKey(0);

            seq = np.array(seq)
            print seq
            findcontiousNum(seq)
            cv2.waitKey(0);
            #print np.array(seq)



