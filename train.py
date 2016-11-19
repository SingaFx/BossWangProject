import numpy as np
import math

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return np.transpose(labels_one_hot)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, 0)
    return np.divide(exp_x, sum_exp_x)

def feed_forward(x, W1, b1, W2, b2):
    h1 = sigmoid(np.add(np.matmul(W1, x), b1))
    y = sigmoid(np.add(np.matmul(W2, h1), b2))
    p = softmax(y)
    return h1, y, p

def weight_variables(dim1, dim2, variance):
    matrix = np.multiply(2, np.random.rand(dim1, dim2))
    matrix = np.multiply(np.subtract(matrix, 1), variance)
    return matrix

def bias_variables(dim1, dim2, variance):
    matrix = np.ones((dim1, dim2))
    matrix = np.multiply(matrix, variance)
    return matrix

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x)) # sigmoid

def get_batch(train_images, train_labels, batch_size=50):
    indexes = np.random.randint(0, train_images.shape[1], batch_size)
    return train_images[:, indexes], train_labels[:, indexes]

def get_batch_auto(train_images, batch_size=50):
    indexes = np.random.randint(0, train_images.shape[1], batch_size)
    return train_images[:, indexes]

def cross_entropy(y, truth):
    entropy = -np.add(np.multiply(truth, np.log(y)), np.multiply(-np.subtract(truth, 1), np.log(-np.subtract(y, 1))))
    mean_entropy = np.sum(entropy) / entropy.shape[1]
    return mean_entropy

def gen_mask(dim1, dim2, dropout):
    mask = np.random.rand(dim1, dim2)
    mask[mask < dropout] = 0
    mask[mask >= dropout] = 1
    return mask

def feed_forward_auto(x, W1, b1, W2, b2):
    h1 = sigmoid(np.add(np.matmul(W1, x), b1))
    y = np.add(np.matmul(W2, h1), b2)
    return h1, y

def evaluate_auto(y, truth):
    # loss = cross_entropy(y, truth)
    loss = (y - truth) **2 / 2
    return loss

def error_rate(y, truth):
    return np.mean(np.argmax(truth, 0) != np.argmax(y, 0))

def evaluate(y, truth):
    loss = cross_entropy(p, truth)
    error = error_rate(p, truth)
    return loss, error

classes = 35
train_images = np.load("pixels.ubyte.npy").T / 255.0
train_labels = dense_to_one_hot(np.load("labels.ubyte.npy"), classes)

np.random.seed(10807)

num_hidden_units = 512
learn_rate = 0.02
epochs = 200000

images = train_images
labels = train_labels

variance1 = math.sqrt(6.0) / (784.0 + num_hidden_units)
W1 = weight_variables(num_hidden_units, 784, variance1)
b1 = bias_variables(num_hidden_units, 1, 0)

variance2 = math.sqrt(6.0) / float(classes + num_hidden_units)
W2 = weight_variables(classes, num_hidden_units, variance2)
b2 = bias_variables(classes, 1, 0)

num_examples = 50
images_ones = np.ones((num_examples, 1))

for epoch in range(0, epochs):
    images, labels = get_batch(train_images, train_labels, num_examples)
    h1, y, p = feed_forward(images, W1, b1, W2, b2)
    images_transposed = np.transpose(images)
    
    h1_tranposed = np.transpose(h1)
    
    if epoch % 5000 == 0:
        train_loss, train_error = evaluate(p, labels)
        print "training loss", train_loss, "training error:", train_error

    loss_derivative = np.subtract(p, labels)
    output = np.multiply(loss_derivative, np.multiply(y, (1-y)))
    hidden = np.multiply(np.matmul(np.transpose(W2), output), np.multiply(h1, (1-h1)))

    W2_gradient = np.matmul(output, h1_tranposed)
    b2_gradient = np.matmul(output, np.ones((h1_tranposed.shape[0], 1)))
    W2 = np.subtract(W2, np.multiply(learn_rate, W2_gradient))
    b2 = np.subtract(b2, np.multiply(learn_rate, b2_gradient))

    W1_gradient = np.matmul(hidden, images_transposed)
    b1_gradient = np.matmul(hidden, images_ones)
    W1 = np.subtract(W1, np.multiply(learn_rate, W1_gradient))
    b1 = np.subtract(b1, np.multiply(learn_rate, b1_gradient))

_, _, p = feed_forward(train_images, W1, b1, W2, b2)
train_loss, train_error = evaluate(p, train_labels)
print "final training loss", train_loss, "training error:", train_error
# _, _, p = feed_forward(test_images, W1, b1, W2, b2)
# test_loss, test_error = evaluate(p, test_labels)
# print "test loss", test_loss, "test error:", test_error

print train_images.shape
print train_labels.shape
