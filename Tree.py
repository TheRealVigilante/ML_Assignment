import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# Define the TNode class for constructing decision tree
class TNode:
    def __init__(self, depth, X, y):
        self.depth = depth
        self.X = X
        self.y = y
        self.left = None
        self.right = None
        self.g = None
        self.j = None
        self.xi = None

    def CalculateGini(self):
        m = len(self.y)
        if m == 0:
            return 0
        class_ratio = np.bincount(self.y) / m
        return 1 - np.sum(class_ratio ** 2)


# Car classification dataset from the previous example
def get_car_data():
    data = {
        'Color': ["Red", "Blue", "Green", "Red", "Green", "Green", "Blue", "Blue", "Red", "Blue", "Green", "Red",
                  "Green", "Green"],
        'Type': ["SUV", "Minivan", "Car", "Minivan", "Car", "SUV", "SUV", "Car", "SUV", "Car", "SUV", "Car", "SUV",
                 "Minivan"],
        'Doors': [2, 4, 4, 4, 2, 4, 2, 2, 2, 4, 4, 2, 2, 4],
        'Tires': ["Whitewall", "Whitewall", "Whitewall", "Blackwall", "Blackwall", "Blackwall", "Blackwall",
                  "Whitewall", "Blackwall", "Blackwall", "Whitewall", "Blackwall", "Blackwall", "Whitewall"],
        'Class': ['Positive', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive',
                  'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Negative']
    }

    df = pd.DataFrame(data)

    # Features and target
    X = df[['Color', 'Type', 'Doors', 'Tires']]  # Independent variables
    y = df['Class'].map({'Positive': 1, 'Negative': 0}).values  # Target variable (map positive/negative to 1/0)

    # Apply One-Hot Encoding to categorical features
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)

    return X_encoded, y


# Function to construct the decision tree recursively
def Construct_Subtree(node, max_depth):
    if node.depth == max_depth or len(node.y) <= 1:
        node.g = np.argmax(np.bincount(node.y))  # Assign the majority class to the leaf
        return
    j, xi = best_split(node)

    # Check if a valid split was found
    if j is None or xi is None:
        node.g = np.argmax(np.bincount(node.y))  # No valid split, make it a leaf node
        return

    node.j = j
    node.xi = xi
    left_indices = node.X[:, j] <= xi
    node.left = TNode(node.depth + 1, node.X[left_indices], node.y[left_indices])
    node.right = TNode(node.depth + 1, node.X[~left_indices], node.y[~left_indices])
    Construct_Subtree(node.left, max_depth)
    Construct_Subtree(node.right, max_depth)


def best_split(node):
    best_gini = node.CalculateGini()
    best_j, best_xi = None, None
    for j in range(node.X.shape[1]):
        for xi in np.unique(node.X[:, j]):
            left_indices = node.X[:, j] <= xi
            left_gini = TNode(0, node.X[left_indices], node.y[left_indices]).CalculateGini()
            right_gini = TNode(0, node.X[~left_indices], node.y[~left_indices]).CalculateGini()
            weighted_gini = (len(node.y[left_indices]) / len(node.y)) * left_gini + (
                        len(node.y[~left_indices]) / len(node.y)) * right_gini
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_j, best_xi = j, xi

    return best_j, best_xi


# Function to predict using the constructed tree
def Predict(X, node):
    if node.g is not None:
        return node.g
    return Predict(X, node.left) if X[node.j] <= node.xi else Predict(X, node.right)


# Plot the decision tree
def plot_tree(node, depth=0, x=0, y=0, dx=1, dy=0.5):
    if node is not None:
        label = f'X[{node.j}] <= {node.xi:.2f}' if node.j is not None else f'Leaf: {node.g}'
        plt.text(x, y, label, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))
        if node.left is not None:
            plot_tree(node.left, depth + 1, x - dx / (2 ** depth), y - dy, dx, dy)
        if node.right is not None:
            plot_tree(node.right, depth + 1, x + dx / (2 ** depth), y - dy, dx, dy)


# Main function to create the decision tree and visualize it
def main():
    X_encoded, y = get_car_data()

    # Initialize the tree root and construct the decision tree
    maxdepth = 5
    treeRoot = TNode(0, X_encoded, y)
    Construct_Subtree(treeRoot, maxdepth)

    # Predict and evaluate the accuracy
    y_hat = np.array([Predict(x, treeRoot) for x in X_encoded])
    accuracy = np.mean(y_hat == y)
    print("Accuracy of the classification tree =", accuracy)

    # Visualize the tree
    plt.figure(figsize=(12, 8))
    plot_tree(treeRoot)
    plt.title('Decision Tree Visualization')
    plt.axis('off')
    plt.show()


# Run the main function
main()
