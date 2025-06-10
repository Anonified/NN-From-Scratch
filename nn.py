from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt


"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
e.g. w_i_h = weights from input layer to hidden layer

"""


def softmax(x):
    exps=np.exp(x-np.max(x))
    return exps/np.sum(exps,axis=0)


images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (50, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 50))
b_i_h = np.zeros((50, 1))
b_h_o = np.zeros((10, 1))

losses=[]
learn_rate = 0.01
epochs = 5
for epoch in range(epochs):
    nr_correct=0
    epoch_loss=0
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o=softmax(o_pre)

        # Cost / Error calculation
        loss = -np.sum(l * np.log(o + 1e-9)) #cross-entropy
        epoch_loss+=loss
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    avg_loss=epoch_loss/len(images)
    losses.append(avg_loss)
    acc = round((nr_correct / len(images)) * 100, 2)
    print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {acc}%, Loss: {avg_loss:.4f}")

#Loss vs Epoch plot
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()


# Show results

n=5
while (n>0):
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))
    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    pred_labels = o_pre.argmax(axis=1)
    misclassified = np.where(pred_labels != o)[0]
    print(f"Total misclassified: {len(misclassified)}")

    plt.title(f"Predicted Digit: {o.argmax()} :)")
    plt.show()
    n-=1
