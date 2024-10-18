import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Define the dataset
data = {
    'X1': [3, 2, 4, 1, 2, 4, 4],
    'X2': [4, 2, 4, 4, 1, 3, 1],
    'Y': ['Red', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (X1, X2)
X = df[['X1', 'X2']].values

# Target variable (convert Red to 1 and Blue to -1)
df['Y'] = df['Y'].map({'Red': 1, 'Blue': -1})
y = df['Y'].values

# Initialize the Support Vector Classifier (linear kernel)
clf = SVC(kernel='linear')

# Train the classifier
clf.fit(X, y)

# Get the coefficients (w1, w2) and intercept (b)
w1, w2 = clf.coef_[0]
b = clf.intercept_[0]

# Print the equation of the hyperplane
print(f"Equation of the optimal separating hyperplane: {w1:.2f} * X1 + {w2:.2f} * X2 + {b:.2f} = 0")

# Optional: You can also get the support vectors
support_vectors = clf.support_vectors_
print("Support vectors:", support_vectors)
# import pandas as pd
# from sklearn.svm import SVC
#
# # Define the dataset
# data = {
#     'X1': [3, 2, 4, 1, 2, 4, 4],
#     'X2': [4, 2, 4, 4, 1, 3, 1],
#     'Y': ['Red', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']
# }
#
# # Convert to DataFrame
# df = pd.DataFrame(data)
#
# # Features (X1, X2)
# X = df[['X1', 'X2']].values
#
# # Target variable (convert Red to 1 and Blue to -1)
# df['Y'] = df['Y'].map({'Red': 1, 'Blue': -1})
# y = df['Y'].values
#
# # Initialize the Support Vector Classifier (linear kernel)
# clf = SVC(kernel='linear')
#
# # Train the classifier
# clf.fit(X, y)
#
# # Get the coefficients (β1, β2) and intercept (β0)
# beta_1, beta_2 = clf.coef_[0]
# beta_0 = clf.intercept_[0]
#
# # Print the coefficients
# print(f"β0 = {beta_0:.2f}")
# print(f"β1 = {beta_1:.2f}")
# print(f"β2 = {beta_2:.2f}")
