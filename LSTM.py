import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import torch.utils.data as Data
from torch.autograd import Variable
import torch.optim as opt
import matplotlib.pyplot as plt


def covert(safe):
    safe.columns = [
        ['num', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'e1', 'e2', 'e3', 'f1', 'f2',
         'f3', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3', 'i1', 'i2', 'i3', 'j1', 'j2', 'j3', 'k1', 'k2', 'k3', 'l1', 'l2',
         'l3', 'm1', 'm2', 'm3', 'n1', 'n2', 'n3', 'o1', 'o2', 'o3', 'p1', 'p2', 'p3', 'q1', 'q2', 'q3']]

    safe = safe.fillna(method='ffill')
    mask = safe[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3', 'q3']] <= 0.5
    data1 = safe[mask]
    data1['count'] = data1.sum(axis=1)
    data1['numm'] = (17 - data1[
        ['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3',
         'q3']].isnull().sum(axis=1)).astype(int)
    data1['score'] = (data1[['count']].values / data1[['numm']].values)
    list_del = data1[data1[['numm']].values == [2]]
    mask2 = list_del[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3',
                      'q3']] > 0.5
    data2 = list_del[mask2]

    data3 = data2[
        ['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3', 'q3']]

    safe2 = safe.drop(data3.index, axis=0)
    res = pd.concat([safe2, data3], join='outer')
    res.sort_index(inplace=True)
    res2 = res.fillna(method='bfill')
    return res2


def pre(frame, feature):
    safe = pd.read_csv('safe_4frame_train.csv', header=None)
    unsafe = pd.read_csv('unsafe_4frame_train.csv', header=None)
    # new add
    safe = covert(safe)
    unsafe = covert(unsafe)
    # end
    safe = safe[0:(len(safe) // frame) * frame]
    print('safe_shape:', safe.shape, len(safe) // frame)
    unsafe = unsafe[0:(len(unsafe) // frame) * frame]
    print('unsafe_shape:', unsafe.shape, len(unsafe) // frame)

    # combine dataframe
    alldata = pd.concat([unsafe, safe], axis=0)
    print('all_data_shape:', alldata.shape)
    del alldata['num']
    # normalization
    dataa = preprocessing.scale(alldata)
    dataa = dataa.reshape(int(alldata.shape[0] / frame), frame, feature)
    print('all_data_shape_reshape:', dataa.shape)

    # label combine
    # safe data add labels
    zero = pd.DataFrame()
    a_0 = np.zeros(len(unsafe) // frame)
    zero['label'] = a_0
    print('unsafe_label_shape', zero.shape)

    one = pd.DataFrame()
    a_1 = np.ones(len(safe) // frame)
    one['label'] = a_1
    print('safe_label_shape', one.shape)

    alldata_label = pd.concat([zero, one], axis=0)
    print(alldata_label.shape)
    labels = alldata_label['label']

    labels = to_categorical(labels)
    X_train = dataa
    y_train = labels

    safe = pd.read_csv('safe_4frame_test.csv', header=None)
    unsafe = pd.read_csv('unsafe_4frame_test.csv', header=None)
    # new add
    safe = covert(safe)
    unsafe = covert(unsafe)
    # end
    safe = safe[0:(len(safe) // frame) * frame]
    print('safe_shape:', safe.shape, len(safe) // frame)
    unsafe = unsafe[0:(len(unsafe) // frame) * frame]
    print('unsafe_shape:', unsafe.shape, len(unsafe) // frame)

    # combine dataframe
    alldata = pd.concat([unsafe, safe], axis=0)
    print('all_data_shape:', alldata.shape)
    del alldata['num']
    # normalization
    dataaa = preprocessing.scale(alldata)
    dataaa = dataaa.reshape(int(alldata.shape[0] / frame), frame, feature)
    print('all_data_shape_reshape:', dataaa.shape)

    # label combine
    # safe data add labels
    zero = pd.DataFrame()
    a_0 = np.zeros(len(unsafe) // frame)
    zero['label'] = a_0
    print('unsafe_label_shape', zero.shape)

    one = pd.DataFrame()
    a_1 = np.ones(len(safe) // frame)
    one['label'] = a_1
    print('safe_label_shape', one.shape)

    alldata_label = pd.concat([zero, one], axis=0)
    print(alldata_label.shape)
    labelss = alldata_label['label']
    print(labelss)
    labelss = to_categorical(labelss)

    X_test = dataaa
    y_test = labelss

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = pre(4, 51)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

X_train, y_train = X_train.type(torch.DoubleTensor), y_train.type(torch.DoubleTensor)
X_test, y_test = X_test.type(torch.DoubleTensor), y_test.type(torch.DoubleTensor)

train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)

batch_size = 24
num_epoch = 100

train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)


class BiLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        output, (h, c) = self.lstm(x, (h_0, c_0))
        out = output[:, -1, :]

        out = self.fc(out)
        return out


input_size = 51
hidden_size = 24
num_layers = 1
output_size = 2

bilstm = BiLstm(input_size, hidden_size, num_layers, output_size)


def get_accuracy(model, train_data):
    model.eval()

    correct = 0
    total = 0

    for i, (x, y) in enumerate(train_data):
        b_x = x.view(-1, 4, 51)
        b_x = b_x.float()
        y = y.float()
        out = model(b_x)
        label = y.argmax(dim=1)
        pred = out.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += x.shape[0]

    return correct / total


losses = []
iters = []
train_acc_list = []
test_acc_list = []


def train_network(model, train_data, test_data, num_epoch, learning_rate=0.01):
    loss_fn = nn.BCELoss()
    optimizer = opt.Adam(model.parameters(), lr=learning_rate)

    print("Training start\n")
    iter = 0

    for epoch in range(num_epoch):
        train_loss = 0
        train_acc = 0

        model.train()
        for i, (x, y) in enumerate(train_data):
            b_x = x.view(-1, 4, 51)
            b_x = b_x.float()
            y = y.float()
            out = model(b_x)

            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_accuracy(model, train_data)

            # plot
            iters.append(iter)
            iter += 1
            losses.append(loss)
            train_acc_list.append(get_accuracy(model, train_data))
            test_acc_list.append(get_accuracy(model, test_data))

        print(
            "Epoch: {}/{}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch + 1, num_epoch, train_loss / i, train_acc / i))

    print("\nTesting start\n")

    test_acc = 0
    model.eval()

    for i, (x, y) in enumerate(test_data):
        b_x = x.view(-1, 4, 51)
        b_x = b_x.float()
        y = y.float()
        out = model(b_x)
        test_acc += get_accuracy(model, test_data)
    print("Average Accuracy per {} Loaders: {:.5f}".format(i, test_acc / i))


def plot(train, test, iters, loss, show_acc=True, show_plot=True):
    if show_acc:
        print("\nFinal Training Accuracy: {:.3f}".format(train[-1]))
        print("Final Testing Accuracy: {:.3f}".format(test[-1]))

    if show_plot:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title("Loss Curve")
        plt.plot(iters[::24], loss[::24], label="Train", linewidth=4, color="#008C76FF")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.title("Accuracy Curve")
        plt.plot(iters[::24], train[::24], label="Train", linewidth=4, color="#9ED9CCFF")
        plt.plot(iters[::24], test[::24], label="Test", linewidth=4, color="#FAA094FF")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")

        plt.legend(loc="best")
        plt.savefig("loss_and_acc.jpg")
        plt.show()


train_network(bilstm, train_loader, test_loader, num_epoch=num_epoch)
plot(train_acc_list, test_acc_list, iters, losses)
torch.save(bilstm, "Lstm_model.pkl")
