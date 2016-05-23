__author__ = 'yujinke'
import  struct
import numpy as np;
from ctypes import create_string_buffer

def writeMnist(data,rows,cols, path_images = "imt_mnist_training_set.data",path_labels="imt_mnist_training_labels.data"):
    _set,_labels = data;
    magic_nums = 2051;
    num_training = len(_set);
    header_images = [magic_nums,num_training,rows,cols];
    header_labels = [magic_nums,num_training];
    print "x"
    print rows
    print cols


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
    print("training_data:",len(_set[0]));

    for i in range(num_training):
        struct.pack_into(len_img_format,buffer_training_set,offset,*_set[i]);
        offset += struct.calcsize(len_img_format);

    file_training_set = open(path_images,'wb');
    file_training_set.write(buffer_training_set);

    #output images to file


    offset =  0 ;
    struct.pack_into(header_labels_format,buffer_training_labels,offset,*header_labels);
    offset += struct.calcsize(header_labels_format);


    for i in range(num_training):
        struct.pack_into(">B",buffer_training_labels,offset,_labels[i]);
        offset += struct.calcsize(">B")
    file_training_label = open(path_labels,'wb');

    file_training_label.write(buffer_training_labels);

    #output labels to file