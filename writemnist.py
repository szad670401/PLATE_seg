__author__ = 'yujinke'
import  struct
import numpy as np;
from ctypes import create_string_buffer
import cv2

def writeMnist(data,rows,cols, path_images = "imt_mnist_training_set.data",path_labels="imt_mnist_training_labels.data"):
    _set,_labels = data;
    model = 0;
   # print type(set[0])
    print _set.dtype

    if(len(_set[0])>0 and _set[0].dtype == np.float32):
        model = 1;
    magic_nums_trainning = 2051;
    magic_nums_labels = 2049;
    print "model",model

    num_training = len(_set);
    header_images = [magic_nums_trainning,num_training,rows,cols];
    header_labels = [magic_nums_labels,num_training];
    len_img = rows*cols;

    header_images_format = '>IIII';
    header_labels_format = '>II';

    len_img_format = '>'+str(len_img)+'B'

    buffer_training_set = create_string_buffer(4*4 + len_img*num_training);
    buffer_training_labels = create_string_buffer(2*4  +  num_training);
    offset = 0 ;

    struct.pack_into(header_images_format,buffer_training_set,offset,*header_images);
    offset += struct.calcsize(header_images_format);
    print(len_img_format)
    for i in range(num_training):
        if(model == 1):
            byte_type = np.array(_set[i])*255
            byte_type = byte_type.astype(np.uint8).ravel();
        else:
            byte_type = _set[i].ravel();
        struct.pack_into(len_img_format,buffer_training_set,offset,*byte_type);
        offset += struct.calcsize(len_img_format);

    file_training_set = open(path_images,'wb');
    file_training_set.write(buffer_training_set);
    offset =  0 ;
    struct.pack_into(header_labels_format,buffer_training_labels,offset,*header_labels);
    offset += struct.calcsize(header_labels_format);
    for i in range(num_training):
        struct.pack_into(">B",buffer_training_labels,offset,_labels[i]);
        offset += struct.calcsize(">B")
    file_training_label = open(path_labels,'wb');
    file_training_label.write(buffer_training_labels);

    #output labels to file