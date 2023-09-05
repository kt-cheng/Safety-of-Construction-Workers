import matplotlib.pyplot as plt

def plot_loss_and_acc(train, test, iters, loss, show_acc=True, show_plot=True):
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