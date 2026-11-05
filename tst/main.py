#!./.venv/bin/python

import numpy as np


# 1. Activation Function & its Derivative
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    # x here is assumed to already be the output of tanh(z)
    return 1.0 - x**2


# 2. XOR Dataset (Adjusted targets to [-1, 1] for tanh)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Expected Targets mapped to match tanh boundaries
y = np.array([[-1], [1], [1], [-1]])

# 3. Initialize Weights and Biases (Using fixed seed for reproducibility)
np.random.seed(42)

# Input layer (2) -> Hidden layer (2)
W1 = np.random.uniform(-1, 1, size=(2, 2))
b1 = np.zeros((1, 2))

# Hidden layer (2) -> Output layer (1)
W2 = np.random.uniform(-1, 1, size=(2, 1))
b2 = np.zeros((1, 1))

# Hyperparameters (tanh often converges faster, so we can use fewer epochs)
learning_rate = 0.1
epochs = 20000

# 4. Training Loop
for epoch in range(epochs):

    # --- FORWARD PASS ---
    hidden_raw = (X @ W1) + b1
    hidden_output = np.tanh(hidden_raw)

    output_raw = (hidden_output @ W2) + b2
    predicted_output = tanh(output_raw)

    # --- CALCULATE LOSS (MSE) ---
    loss = np.mean(0.5 * (y - predicted_output) ** 2)

    # --- BACKWARD PASS ---
    # Error term (delta) at the output layer
    error_output = predicted_output - y
    delta_output = error_output * tanh_derivative(predicted_output)

    # Error term (delta) passed back to hidden layer
    error_hidden = np.dot(delta_output, W2.T)
    delta_hidden = error_hidden * tanh_derivative(hidden_output)

    # --- GRADIENTS & PARAMETER UPDATES ---
    W2 -= learning_rate * np.dot(hidden_output.T, delta_output)
    b2 -= learning_rate * np.sum(delta_output, axis=0, keepdims=True)

    W1 -= learning_rate * np.dot(X, delta_hidden)
    b1 -= learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    # Print status every 4,000 iterations
    if epoch % 4000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# 5. Final Evaluation
print("\n--- Final Predictions After Training ---")
final_hidden = tanh(np.dot(X, W1) + b1)
final_predictions = tanh(np.dot(final_hidden, W2) + b2)

for inputs, pred, target in zip(X, final_predictions, y):
    print(f"Input: {inputs} -> Predicted: {pred[0]:.4f} (Target: {target[0]})")
