import torch.nn as nn
import torch.optim as opt

from get_accuracy import get_accuracy

losses = []
iters = []
train_acc_list = []
test_acc_list = []

def train_network(model, train_data, test_data, num_epoch, learning_rate=0.01):
    loss_fn = nn.BCELoss()
    optimizer = opt.Adam(model.parameters(), lr=learning_rate)

    print("*******Training start*******\n")
    
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

    print("\n*******Testing start*******\n")

    test_acc = 0
    model.eval()

    for i, (x, y) in enumerate(test_data):
        b_x = x.view(-1, 4, 51)
        b_x = b_x.float()
        y = y.float()
        out = model(b_x)
        test_acc += get_accuracy(model, test_data)
    print("Average Accuracy per {} Loaders: {:.5f}".format(i, test_acc / i))
    
    return train_acc_list, test_acc_list, iters, losses