This GitHub repository contains a Python script that implements the Principal Component Analysis (PCA) algorithm to reduce the dimensionality of a dataset and provides a graphical user interface (GUI) for users to interact with the algorithm. PCA is a dimensionality reduction technique that is widely used in machine learning and data analysis.

# Contents
## Main Script: The main script is named algorithm.py and contains the PCA algorithm implementation. It includes the following key components:
Importing necessary libraries such as scikit-learn, tkinter, numpy, pandas, matplotlib, and more.
Functions for performing PCA with different numbers of components (1, 2, and 3).
A GUI created with tkinter, which allows the user to select the number of components for PCA and visualize the results.

## Background Image: The GUI includes a background image (named asset.jpg) to enhance the visual appearance of the interface.

## Dataset: The script uses the Iris dataset, which is loaded from an external URL (UCI Machine Learning Repository).

## Usage
To use this PCA implementation with the GUI, follow these steps:

1. Clone or download the repository to your local machine.

2. Ensure you have the required Python libraries installed. You can typically install them using pip:

  ```sh
  pip install scikit-learn pandas matplotlib pillow plotly

3. Run the script pca_gui.py using a Python interpreter. This will launch the GUI.

3. In the GUI, you can choose to perform PCA with 1, 2, or 3 components.

4. The script will load the Iris dataset, apply PCA, and display a confusion matrix and accuracy score for the selected number of components. It will also display a scatter plot or 3D scatter plot to visualize the data's reduced dimensionality.


# References
PCA (Principal Component Analysis) - Official scikit-learn documentation on PCA.
Iris Dataset - UCI Machine Learning Repository link for the Iris dataset.
