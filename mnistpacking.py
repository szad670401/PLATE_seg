__author__ = 'yujinke'

import numpy as np
import struct
import matplotlib.pyplot as plt

filename = 'train-images.idx3-ubyte'
filename_labels = 'train-labels.idx1-ubyte'


binfile = open(filename , 'rb')
binlabels  = open(filename_labels,'rb')
buf = binfile.read()
buflabels = binlabels.read();

index = 0 ;
index_l = 0 ;

magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
magic_l,numImages_l = struct.unpack_from('>II',buflabels,index_l);
index += struct.calcsize('>IIII')
index_l += struct.calcsize('>II')


print numImages,numImages_l
print magic,numImages,numRows,numColumns;

for i in range(numImages):
    im = struct.unpack_from('>1296B' ,buf,index)
    val= struct.unpack_from('>1B',buflabels,index_l);

    print val;
    index += struct.calcsize('>1296B')
    index_l += struct.calcsize('>1B')


    print im
    im = np.array(im)
    im = im.reshape(36,36)

    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im , cmap='gray')
    plt.show()



