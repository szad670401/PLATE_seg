__author__ = 'yujinke'

import numpy as np

import  cv2;

import cPickle


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
        self.model
    def predict(self, samples):
        return self.model.predict_all(samples).ravel()

    def predict_single(self,sample):
        return self.predict(np.array([sample]));

# model = SVM(C=1.67, gamma=0.0383)
# model.load('digits_svm.dat')
# f = open("./data.pkl", 'rb')
# training_data, validation_data, test_data = cPickle.load(f)
# samples_train, labels_train= test_data
#
# print "result",model.predict_single(samples_train[1]),labels_train[1]
# print len(samples_train[1])
#




