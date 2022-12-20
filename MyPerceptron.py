import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


def plot_digit(imgdata):
    '''
    Draw mnist image
    :param imgdata: imgdata is a numpy array
    :return:
    '''
    image = imgdata.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def load_mnist01():
    '''
    This function is used to generate 0 and 1 images from the mnist
    :return:
    '''
    # Scikit-Learn加载的数据集通常具有类似的字典结构，包括：
    # DESCR键，描述数据集
    # data键，包含一个数组，每个实例为一行，每个特征为一列
    # target键，一个标签的数组。
    mnist = fetch_openml("mnist_784", data_home='./mnist')
    data, label = mnist["data"], mnist["target"]
    # data.shape: (70000, 784) label.shape (70000,)
    print('datatype:', type(data), 'labeltype', type(label))
    print('data.shape:', data.shape, 'label.shape', label.shape)

    # 取两类
    mnistimg = np.array(data.values)
    label = np.array(label.values)

    zeros_index = np.squeeze(np.where(label == '0'))
    ones_index = np.squeeze(np.where(label == '1'))
    zeros_img = mnistimg[zeros_index]
    ones_img = mnistimg[ones_index]
    print('num 0：', len(zeros_img), 'num 1：', len(ones_img))  # 0的个数： 6903 1的个数： 7877

    mnistimg_01 = np.vstack((zeros_img, ones_img))
    label_01 = np.array([0] * len(zeros_img) + [1] * len(ones_img)).astype(np.uint8)
    label_01 = label_01.reshape(-1, 1)

    train_img, test_img, train_label, test_label = train_test_split(mnistimg_01, label_01, test_size=0.2, random_state=0)

    # 数据归一化，加快学习过程，防止某些情况下训练过程出现计算溢出
    train_img = train_img.astype(float) / 255.0
    train_label = train_label.astype(float)
    test_img = test_img.astype(float) / 255.0
    test_label = test_label.astype(float)
    print('train.shape', train_img.shape, 'train_label', train_label.shape)

    print('Data generation completed!')
    return train_img, test_img, train_label, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class perceptron(object):
    def __init__(self, num_of_weights):
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)  # 使用np.random.randn随机生成一个 num_of_weights*1 的列向量，该向量即为权值W
        self.b = 0.

    def forward(self, x):  # 加权求和单元和非线性函数单元通过定义计算过程来实现
        z = np.dot(x, self.w) + self.b  # 加权求和
        pred_y = sigmoid(z)  # 非线性函数sigmoid
        return pred_y

    def loss_fun(self, pred_y, true_y):
        """
        pred_y：网络对一批样本的预测值组成的列向量
        true_y：一批样本的真实标签
        """
        error = pred_y - true_y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def evaluate(self, pred_y, true_y, threshold=0.5):
        pred_y[pred_y < threshold] = 0  # 预测值小于0.5，则判为类别0
        pred_y[pred_y >= threshold] = 1

        acc = np.mean((pred_y == true_y).astype(float))
        return acc

    def gradient(self, x, y, pred_y):
        gradient_w = (pred_y - y) * pred_y * (1 - pred_y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (pred_y - y) * pred_y * (1 - pred_y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, lr=0.01):
        self.w = self.w - lr * gradient_w
        self.b = self.b - lr * gradient_b

    def train(self, train_x, train_y, test_x, test_y, max_epochs=100, lr=0.01):
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        for epoch in range(1, max_epochs + 1):
            pred_y_train = self.forward(train_x)
            gradient_w, gradient_b = self.gradient(train_x, train_y, pred_y_train)
            self.update(gradient_w, gradient_b, lr)
            if (epoch == 1) or (epoch % 200 == 0):
                pred_y_test = self.forward(test_x)
                train_loss = self.loss_fun(pred_y_train, train_y)
                test_loss = self.loss_fun(pred_y_test, test_y)
                train_acc = self.evaluate(pred_y_train, train_y)
                test_acc = self.evaluate(pred_y_test, test_y)
                print('epoch: %d, train_loss: %.4f, test_loss: %.4f, train_acc: %.4f, test_acc: %.4f' % (
                epoch, train_loss, test_loss, train_acc, test_acc))
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
        return train_losses, test_losses, train_accs, test_accs


def plot_metric(train_loss, test_loss, train_acc, test_acc, max_epochs):
    # 画出各指标的变化趋势
    plot_x = np.arange(0, max_epochs + 1, 200)
    plot_y_1 = np.array(train_loss)
    plot_y_2 = np.array(test_loss)
    plot_y_3 = np.array(train_acc)
    plot_y_4 = np.array(test_acc)
    plt.plot(plot_x, plot_y_1, color='r', linestyle='--', label='train_loss')
    plt.plot(plot_x, plot_y_2, color='g', linestyle='-', label='test_loss')
    plt.plot(plot_x, plot_y_3, color='r', linestyle='--', label='train_acc')
    plt.plot(plot_x, plot_y_4, color='g', linestyle='-', label='test_acc')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.grid()
    plt.savefig('./perceptron.png')
    plt.show()


def main():
    train_img, test_img, train_label, test_label = load_mnist01()

    start_time = time.time()
    # 创建网络
    selfperceptron = perceptron(28 * 28)
    max_epochs = 5000
    # 启动训练
    train_loss, test_loss, train_acc, test_acc = selfperceptron.train(train_img, train_label, test_img, test_label,
                                                                  max_epochs=max_epochs, lr=0.01)
    print('cost time: %.1f s' % (time.time() - start_time))

    plot_metric(train_loss, test_loss, train_acc, test_acc, max_epochs)


if __name__ == '__main__':
    main()

