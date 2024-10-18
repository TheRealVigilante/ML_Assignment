import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

# Sample dataset
data = {
    'Color': ["Red", "Blue", "Green", "Red", "Green", "Green", "Blue", "Blue", "Red", "Blue", "Green", "Red", "Green",
              "Green"],
    'Type': ["SUV", "Minivan", "Car", "Minivan", "Car", "SUV", "SUV", "Car", "SUV", "Car", "SUV", "Car", "SUV",
             "Minivan"],
    'Doors': [2, 4, 4, 4, 2, 4, 2, 2, 2, 4, 4, 2, 2, 4],
    'Tires': ["Whitewall", "Whitewall", "Whitewall", "Blackwall", "Blackwall", "Blackwall", "Blackwall", "Whitewall",
              "Blackwall", "Blackwall", "Whitewall", "Blackwall", "Blackwall", "Whitewall"],
    'Class': ['Positive', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive',
              'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Negative']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Color', 'Type', 'Doors', 'Tires']]  # Independent variables
y = df['Class']  # Target variable (positive/negative)

# Apply One-Hot Encoding for categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)


# Function to calculate entropy
def entropy(y):
    class_counts = np.bincount(y == 'Positive')  # Count positive and negative examples
    probabilities = class_counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


# Function to calculate information gain
def information_gain(X, y, feature_index):
    # Calculate the original entropy
    original_entropy = entropy(y)

    # Get unique values in the feature
    values, counts = np.unique(X[:, feature_index], return_counts=True)

    # Calculate the weighted average entropy after the split
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset_y = y[X[:, feature_index] == value]
        weighted_entropy += (count / len(y)) * entropy(subset_y)

    # Information gain is the difference in entropy before and after the split
    return original_entropy - weighted_entropy


# Calculate the initial entropy
initial_entropy = entropy(y)
print(f"Initial entropy of the dataset: {initial_entropy:.4f}")

# Calculate the information gain for each feature
for i, feature in enumerate(encoder.get_feature_names_out(X.columns)):
    gain = information_gain(X_encoded, y, i)
    print(f"Information Gain for {feature}: {gain:.4f}")
