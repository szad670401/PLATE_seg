__author__ = 'yujinke'


import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU
import cPickle
training_data, validation_data, test_data= network3.load_data_unshared("./data.pkl")
mini_batch_size = 10

# net = Network([
#         ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                       filter_shape=(20, 1, 5, 5),
#                       poolsize=(2, 2),
#                       activation_fn=ReLU),
#         ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
#                       filter_shape=(40, 20, 5, 5),
#                       poolsize=(2, 2),
#                       activation_fn=ReLU),
#         FullyConnectedLayer(
#             n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
#         FullyConnectedLayer(
#             n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
#         SoftmaxLayer(n_in=1000, n_out=31, p_dropout=0.5)],
#         mini_batch_size)
#
# net = Network([
#         ConvPoolLayer(image_shape=(mini_batch_size, 1, 48, 48),
#                       filter_shape=(25, 1, 5, 5),
#                       poolsize=(2, 2),
#                       activation_fn=ReLU),
#         ConvPoolLayer(image_shape=(mini_batch_size, 25, 22, 22),
#                       filter_shape=(16, 25, 5, 5),
#                       poolsize=(2, 2),
#                       activation_fn=ReLU),
#         FullyConnectedLayer(n_in=16*4*4, n_out=120, activation_fn=ReLU),
#          FullyConnectedLayer(n_in=120, n_out=84, activation_fn=ReLU),
#         SoftmaxLayer(n_in=84, n_out=36)], mini_batch_size )
#


net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(25, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 25, 12, 12),
                      filter_shape=(40, 25, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
    FullyConnectedLayer(n_in=100, n_out=500, activation_fn=ReLU),
        SoftmaxLayer(n_in=500, n_out=36)], mini_batch_size )


net.SGD(training_data,20, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
# training_data1, validation_data1, test_data1 = cPickle.load(open("./data.pkl", 'rb'))
# training_x, training_y = training_data1;
#
#
#
# print(len(training_x[0]))
#
#
# net.predict(training_x[0])

cPickle.dump(net,open("./weights.pkl","wb"))

