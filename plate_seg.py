#coding=utf-8
__author__ = 'yujinke'
from sklearn import preprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from multiprocessing.pool import ThreadPool
from numpy.linalg import norm


dir_eng = "./chars2"
dir_ch = "./charsChinese"
dir_num = "./chars3"

SZ =  20


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess_hog(digits):
        gx = cv2.Sobel(deskew(digits), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(deskew(digits), cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        return hist;









set  = []
labels = []


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()


def evaluate_model(model,  samples, labels):

    resp = model.predict(samples)
    print resp;

    err = (labels != resp).mean()
    print 'error: %.2f %%' % (err*100)




def findinside(dirname,code):
    for parent,dirnames,filenames in os.walk(dir_eng):
        for filename in filenames:
            path  =parent+"/"+filename
            if(path.endswith(".jpg")):
                        img = cv2.imread(path,cv2.CV_LOAD_IMAGE_GRAYSCALE);
                        fature =  preprocess_hog(img);
                        set.append(fature);
                        labels.append(code);




for parent,dirnames,filenames in os.walk(dir_eng):

        for dirname in range(len(dirnames)):
                    c_path =  dir_eng+"/"+ dirnames[dirname];
                    findinside(c_path,dirname);




shuffle = np.random.permutation(len(set));
set = np.array(set);
labels = np.array(labels);

set, labels = set[shuffle], labels[shuffle]

print set.shape


train_n = int(0.9*len(set))


print labels

digits_train, digits_test = np.split(set, [train_n])
samples_train, samples_test = np.split(labels, [train_n])

digits_train.dtype = np.float32
digits_test.dtype = np.float32

model = SVM(C=2.67, gamma=5.383);

model.train(digits_train, samples_train)
vis = evaluate_model(model,  digits_test, samples_test)



