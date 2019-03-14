from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
learning_rate = 0.1

for i in range(10000):
    print('Try: ' + str(i + 1))
    grads = network.gradient(x_train, t_train)
    for key in network.params:
        network.params[key] -= learning_rate * grads[key]
    print('Accuracy: ' + str(network.accuracy(x_train, t_train)) + '\n')
