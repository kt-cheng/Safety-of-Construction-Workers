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