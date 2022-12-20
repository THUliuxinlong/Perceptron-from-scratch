import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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

    train_img, test_img, train_label, test_label = train_test_split(mnistimg_01, label_01, test_size=0.2, random_state=0)

    # 数据归一化，加快学习过程，防止某些情况下训练过程出现计算溢出
    train_img = train_img.astype(float) / 255.0
    train_label = train_label.astype(float)
    test_img = test_img.astype(float) / 255.0
    test_label = test_label.astype(float)
    print('train.shape', train_img.shape, 'train_label', train_label.shape)

    print('Data generation completed!')
    return train_img, test_img, train_label, test_label


def svm_test(train_img, test_img, train_label, test_label, kernel):
    svm_linear = SVC(kernel=kernel)
    start_time = time.time()
    svm_linear.fit(train_img, train_label)
    print('cost time: %.1f s' % (time.time() - start_time))

    test_pred = svm_linear.predict(test_img)
    test_acc = accuracy_score(test_label, test_pred)
    print(f'kernel={kernel} acc:', test_acc)


def main():
    train_img, test_img, train_label, test_label = load_mnist01()

    svm_test(train_img, test_img, train_label, test_label, kernel='linear')
    svm_test(train_img, test_img, train_label, test_label, kernel='poly')
    svm_test(train_img, test_img, train_label, test_label, kernel='rbf')
    svm_test(train_img, test_img, train_label, test_label, kernel='sigmoid')


if __name__ == '__main__':
    main()

