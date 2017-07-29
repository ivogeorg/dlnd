import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
# epochs = 900
epochs = 1
learnrate = 0.005

print("DEBUG: INI ------------")
print("\nDEBUG: input: features shape={}".format(features.shape))
print("\nDEBUG: input: targets shape={}".format(targets.shape))
print("\nDEBUG: input: features test shape={}".format(features_test.shape))
print("\nDEBUG: input: targets test shape={}".format(targets_test.shape))
n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)
print("\nDEBUG: ini: i2h shape={}".format(weights_input_hidden.shape))
print("\nDEBUG: ini: i2h=\n{}".format(weights_input_hidden))
print("\nDEBUG: ini: h2o shape={}".format(weights_hidden_output.shape))
print("\nDEBUG: ini: h2o=\n{}".format(weights_hidden_output))

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    print("\nDEBUG: var: del i2h shape={}".format(del_w_input_hidden.shape))
    print("\nDEBUG: var: del i2h=\n{}".format(del_w_input_hidden))
    print("\nDEBUG: var: del h2o shape={}".format(del_w_hidden_output.shape))
    print("\nDEBUG: var: del h2o=\n{}".format(del_w_hidden_output))

    for x, y in zip(features.values[:2], targets[:2]):
        print("\nDEBUG: x={}".format(x))
        print("\nDEBUG: y={}".format(y))

        ## Forward pass ##
        print("\nDEBUG: FWD ------------")

        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        print("\nDEBUG: x • i2h = hidden_input =>\n{}\n\n{}\n\n{}".format(
            x,
            weights_input_hidden,
            hidden_input
        ))

        hidden_output = sigmoid(hidden_input)
        print("\nDEBUG: sig(hidden_input) = hidden_output =>\n{}\n\n{}".format(
            hidden_input,
            hidden_input
        ))

        output = sigmoid(np.dot(hidden_output,
                                weights_hidden_output))
        print("\nDEBUG: sig(hidden_output • h20) = output =>\n{}\n\n{}\n\n{}".format(
            hidden_output,
            weights_hidden_output,
            output
        ))

        ## Backward pass ##
        print("\nDEBUG: BCK ------------")

        # TODO: Calculate the network's prediction error
        error = y - output
        print("\nDEBUG: y - output = error =>\n{}\n\n{}\n\n{}".format(
            y,
            output,
            error
        ))

        # TODO: Calculate error term for the output unit
        output_error_term = error * output * (1 - output)
        print("\nDEBUG: error * output * (1-output) = output_error_term =>\n{}\n\n{}\n{}\n\n{}".format(
            error,
            output,
            1-output,
            output_error_term
        ))

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)
        print("\nDEBUG: output_error_term • weights_hidden_output = hidden_error =>\n{}\n\n{}\n\n{}".format(
            output_error_term,
            weights_hidden_output,
            hidden_error
        ))

        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)
        print("\nDEBUG: hidden_error * hidden_output * (1-hidden_output) = hidden_error_term =>\n{}\n\n{}\n{}\n\n{}".format(
            hidden_error,
            hidden_output,
            1-hidden_output,
            hidden_error_term
        ))

        # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        print("\nDEBUG: output_error_term * hidden_output =+ del_w_hidden_output =>\n{}\n\n{}\n\n{}".format(
            output_error_term,
            hidden_output,
            del_w_hidden_output
        ))

        del_w_input_hidden += hidden_error_term * x[:, None]
        print("\nDEBUG: hidden_error_term * x[:, None] =+ del_w_input_hidden =>\n{}\n\n{}\n\n{}".format(
            hidden_error_term,
            x[:, None],
            del_w_input_hidden
        ))

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    print("\nDEBUG: learnrate * del_w_input_hidden / n_records =+ weights_input_hidden =>\n{}\n\n{}\n\n{}\n\n{}".format(
        learnrate,
        del_w_input_hidden,
        n_records,
        weights_input_hidden
    ))

    weights_hidden_output += learnrate * del_w_hidden_output / n_records
    print("\nDEBUG: learnrate * del_w_hidden_output / n_records =+ weights_hidden_output =>\n{}\n\n{}\n\n{}\n\n{}".format(
        learnrate,
        del_w_hidden_output,
        n_records,
        weights_hidden_output
    ))

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
