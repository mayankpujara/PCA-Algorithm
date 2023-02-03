# Importing Libraries

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tkinter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import plotly.express as px
from PIL import ImageTk, Image

# Function for 1 Component PCA


def PCA_1():
    from sklearn.decomposition import PCA
    pca_1 = PCA(n_components=1)
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = pca_1.fit_transform(X_train)
    X_test = pca_1.transform(X_test)
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\nFor 1 Component PCA\n")
    print("Confusion Matrix is:\n", cm)
    print('\nAccuracy: ' + str(accuracy_score(y_test, y_pred)*100)+' %')
    df = px.data.iris()
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    pca_1 = PCA(n_components=1)
    components = pca_1.fit_transform(X)
    total_var = pca_1.explained_variance_ratio_.sum() * 100
    fig = px.scatter(components, x=0, color=df['species'],
                     title=f'Total Explained Variance: {total_var:.2f}%')
    fig.show()

# Function for 2 Component PCA


def PCA_2():
    from sklearn.decomposition import PCA
    pca_2 = PCA(n_components=2)
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = pca_2.fit_transform(X_train)
    X_test = pca_2.transform(X_test)

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("\nFor 2 Component PCA\n")
    print("Confusion Matrix is:\n", cm)
    print('\nAccuracy: ' + str(accuracy_score(y_test, y_pred)*100)+' %')

    df = px.data.iris()
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    pca_2 = PCA(n_components=2)
    components = pca_2.fit_transform(X)
    total_var = pca_2.explained_variance_ratio_.sum() * 100
    fig = px.scatter(components, x=0, y=1, color=df['species'],
                     title=f'Total Explained Variance: {total_var:.2f}%')
    fig.show()

# Function for 3 Component PCA


def PCA_3():
    from sklearn.decomposition import PCA
    pca_3 = PCA(n_components=3)
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = pca_3.fit_transform(X_train)
    X_test = pca_3.transform(X_test)

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("\nFor 3 Component PCA\n")
    print("Confusion Matrix is:\n", cm)
    print('\nAccuracy: ' + str(accuracy_score(y_test, y_pred)*100)+' %')

    df = px.data.iris()
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    pca_3 = PCA(n_components=3)
    components = pca_3.fit_transform(X)

    total_var = pca_3.explained_variance_ratio_.sum() * 100

    fig = px.scatter_3d(
        components, x=0, y=1, z=2, color=df['species'],
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()


# Created GUI using tkinter
window = Tk()

# Adding a background image to the GUI
bg = ImageTk.PhotoImage(Image.open('background.jpg'))
background = Label(window, image=bg)
background.place(x=0, y=0)

title = Label(window, text="Implementing PCA Algorithm",
              bg="black", fg="lightblue", font=("Times New Roman", 16))
title.place(x=80, y=40)
title.pack(expand=True)

button1 = Button(window, text="Containing 1 component", width=25,
                 height=2, bg="lightblue", fg="black", command=PCA_1)
button1.place(x=80, y=120)
button1.pack(expand=True)

button2 = Button(window, text="Containing 2 components", width=25,
                 height=2, bg="lightblue", fg="black", command=PCA_2)
button2.place(x=80, y=140)
button2.pack(expand=True)

button3 = Button(window, text="Containing 3 components", width=25,
                 height=2, bg="lightblue", fg="black", command=PCA_3)
button3.place(x=80, y=160)
button3.pack(expand=True)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"  # Dataset
names = ['sepal-length', 'sepal-width',
         'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)

window.title('ML Experiment No. 8 (Mayank Pujara)')
window.geometry("500x450+10+10")
window.mainloop()
