# Lab Exercise 1: PCA
# Name: Dikshith reddy Macherla
# Student Number: 0789055

# Import necessary packages
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import imageio
import numpy as np
from skimage import color

# Part 2: Loading the Iris Data and Creating Plots
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Class labels
target_names = iris.target_names

# Scatter plot for the first two variables, colored by class
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel('Feature 1: Sepal Length (cm)')
plt.ylabel('Feature 2: Sepal Width (cm)')
plt.title('Scatterplot of Iris Data\nDikshith Reddy M - 0789055')
plt.legend()
plt.savefig('scatterplot_iris.png')  # Save the figure
plt.show()

# Another type of plot (e.g., histogram of feature distributions)
plt.figure(figsize=(8, 6))
plt.hist(X[:, 2], bins=15, alpha=0.7, label='Petal Length')
plt.hist(X[:, 3], bins=15, alpha=0.7, label='Petal Width')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Features\nDikshith Reddy M - 0789055')
plt.legend()
plt.savefig('histogram_iris.png')  # Save the figure
plt.show()

# Part 3: Principal Component Analysis (PCA)
# PCA on an Image
def run_pca_on_image(image_path, number_components=20):
    # Load and display the original image
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    # Convert the image to grayscale
    img_gray = color.rgb2gray(img)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()

    # Apply PCA
    pca = PCA(number_components)
    img_transformed = pca.fit_transform(img_gray)
    img_inverted = pca.inverse_transform(img_transformed)

    # Display the PCA-reconstructed image
    plt.imshow(img_inverted, cmap='gray')
    plt.title(f'PCA Reconstructed Image with {number_components} Components')
    plt.axis('off')
    plt.show()

    # Display the explained variance ratio
    comps = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
    print(f'Explained Variance Ratios (%): {comps}')

image_path = 'tree.jpeg'
run_pca_on_image(image_path, number_components=100)
