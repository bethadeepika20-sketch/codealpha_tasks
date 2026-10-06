# Iris Flower Classification

## Project Overview

This project is completed as part of the CodeAlpha Data Science Internship.

The objective of this project is to build a machine learning classification model that identifies the species of an Iris flower based on its measurements.

## Dataset

The Iris dataset contains 150 flower samples belonging to three species:

* Setosa
* Versicolor
* Virginica

The features used for classification are:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Workflow

1. Load the Iris dataset
2. Explore the dataset
3. Check missing values
4. Perform exploratory data analysis
5. Create visualizations
6. Split the data into training and testing sets
7. Train a Random Forest Classifier
8. Make predictions
9. Evaluate the model
10. Visualize the confusion matrix

## Machine Learning Model

A Random Forest Classifier was used to predict the Iris flower species.

The dataset was divided into:

* 80% training data
* 20% testing data

The model was trained using 100 decision trees.

## Results

The Random Forest Classifier achieved an accuracy of **90.00%** on the test dataset.

### Classification Performance

| Species    | Precision | Recall | F1-score |
| ---------- | --------: | -----: | -------: |
| Setosa     |      1.00 |   1.00 |     1.00 |
| Versicolor |      0.82 |   0.90 |     0.86 |
| Virginica  |      0.89 |   0.80 |     0.84 |

## Visualizations

The project includes:

* Iris species distribution
* Petal length comparison by species
* Confusion matrix

## Conclusion

This project demonstrates how machine learning can be used to classify Iris flower species based on their physical measurements. The Random Forest Classifier achieved 90% accuracy on the test data.

## Internship

Completed as part of the **CodeAlpha Data Science Internship**.
