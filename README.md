# PCA (Principal Component Analysis) Algorithm 
This GitHub repository contains a Python script that implements the Principal Component Analysis (PCA) algorithm to reduce the dimensionality of a dataset and provides a graphical user interface (GUI) for users to interact with the algorithm. PCA is a dimensionality reduction technique that is widely used in machine learning and data analysis.

# Prerequisites

Before using this code, you should have the following installed:

- Python (>=3.7)
  
# Contents
### Main Script: 
The main script is named algorithm.py and contains the PCA algorithm implementation. It includes the following key components:
1. Importing necessary libraries such as scikit-learn, tkinter, numpy, pandas, matplotlib, and more.
2. Functions for performing PCA with different numbers of components (1, 2, and 3).
3. A GUI created with tkinter, which allows the user to select the number of components for PCA and visualize the results.

### Background Image: 
The GUI includes a background image (named asset.jpg) to enhance the visual appearance of the interface.

### Dataset: 
The script uses the Iris dataset, which is loaded from an external URL (UCI Machine Learning Repository).

You can access the dataset [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

### Usage
To use this PCA implementation with the GUI, follow these steps:
1. Clone or download the repository to your local machine.
   ```sh
   git clone https://github.com/your_username/PCA-Algorithm.git
2. Ensure you have the required Python libraries installed. You can typically install them using pip:
   ```sh
   pip install scikit-learn pandas matplotlib pillow plotly

3. Run the script algorithm.py using a Python interpreter. This will launch the GUI.
4. In the GUI, you can choose to perform PCA with 1, 2, or 3 components.
5. The script will load the Iris dataset, apply PCA, and display a confusion matrix and accuracy score for the selected number of components. It will also display a scatter plot or 3D scatter plot to visualize the data's reduced dimensionality.
