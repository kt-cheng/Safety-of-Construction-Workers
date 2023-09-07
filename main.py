import torch
import torch.utils.data as Data

from bilstm import BiLstm
from pre_processing import pre_processing
from train_network import train_network
from plot import plot_loss_and_acc

x_train, y_train = pre_processing(train=True, frame=4, feature=51)
x_test, y_test = pre_processing(train=False, frame=4, feature=51)

X_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

X_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

X_train, y_train = X_train.type(torch.DoubleTensor), y_train.type(torch.DoubleTensor)
X_test, y_test = X_test.type(torch.DoubleTensor), y_test.type(torch.DoubleTensor)

train_dataset = Data.TensorDataset(X_train, y_train)
test_dataset = Data.TensorDataset(X_test, y_test)

batch_size = 24
num_epoch = 100

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

input_size = 51
hidden_size = 24
num_layers = 1
output_size = 2

bilstm = BiLstm(input_size, hidden_size, num_layers, output_size)

train_acc_list, test_acc_list, iters, losses = train_network(bilstm, train_loader, test_loader, num_epoch=num_epoch)
plot_loss_and_acc(train_acc_list, test_acc_list, iters, losses)
torch.save(bilstm, "Lstm_model.pkl")
