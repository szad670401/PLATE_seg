__author__ = 'yujinke'


import os
import cv2
import numpy as np

import writemnist
import cPickle
import gzip
import matplotlib as plt
import rotrandom


chars = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]

size =[28,28]
reverse = True;

def make_dataset(dirn):
    set = [];
    labels = [] ;


    def findinside(dirname,code,):
        print "code",code;
        print "dirname",dirname;


        for parent,dirnames,filenames in os.walk(dirname):
            adder = 2000 -  len(filenames)
            len_d = len(filenames)
            for filename in filenames:
                path  =parent+"/"+filename
                if(path.endswith(".jpg") or path.endswith(".png")):
                            img = cv2.imread(path,cv2.CV_LOAD_IMAGE_GRAYSCALE);
                            if(reverse == True):
                                img = cv2.bitwise_not(img)


                            img = cv2.resize(img,(size[0],size[1]));
                            img = img.astype(np.float32)/255;


                            set.append(img.ravel());
                            labels.append(code);
            for i in range(adder):
                c_index = int(np.random.rand() * len_d);
                l_set = len(set)
                set.append(rotrandom.rotRandrom( set[l_set-len_d + c_index],1,(size[0],size[1])));

                labels.append(code);

        print len(set),dirname,len(filenames)

    for parent,dirnames,filenames in os.walk(dirn):
            num = len(dirnames);
            for i in range(num):
                        c_path = dir_chars + "/"+ dirnames[i];
                        findinside(c_path,i);


    shuffle = np.random.permutation(len(set));

    print len(set)
    set = np.array(set);
    labels = np.array(labels);
    set, labels = set[shuffle], labels[shuffle]
    train_n = int(0.9*len(set))

    training_set,test_set = np.split(set, [train_n])
    training_labels, test_labels = np.split(labels, [train_n])
    print training_labels
    validation_set = test_set.copy();
    validation_labels = test_set.copy();
    training_data = [training_set,training_labels]
    validation_data = [validation_set,validation_labels]

    test_data = [test_set,test_labels]

    data = [ training_data, validation_data, test_data];
    fileid  = open("./data.pkl","wb")
    cPickle.dump(data,fileid)



def display():

    X   =  cPickle.load(open("./data.pkl","rb"))
    training_data,valiation_Data,test_data = X
    imgs,labels = training_data;
    imgs_test,labels_test = test_data;
    print len(imgs_test)


    x = 0;
    writemnist.writeMnist(training_data,size[0],size[1],"./train-images.idx1-ubyte","./train-labels.idx3-ubyte")
    writemnist.writeMnist(test_data,size[0],size[1],"./t10k-images.idx3-ubyte","./t10k-labels.idx1-ubyte")

    #
    # for img in imgs:
    #     img =  img.reshape(28,28)
    #     cv2.imshow("img",img);
    #     print chars[labels[x]];
    #     cv2.waitKey(0);
    #
    #     x+=1
    #


dir_chars = "./Fnt"

#dir_chars = "/Users/yujinke/learning cpp/PLATE_seg/class"
make_dataset(dir_chars);
#display()




