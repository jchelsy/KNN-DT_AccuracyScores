# Import Libraries
from math import e
import numpy as np
import pandas as pd
from sklearn import tree  # DT lib
from sklearn.neighbors import KNeighborsClassifier  # KNN lib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import openpyxl
import warnings  # ignore warnings
warnings.filterwarnings("ignore")


################################################

# DATA FROM:
# https://www.kaggle.com/datasets/andrewmvd/early-diabetes-classification

"""
age                     -   Patient age
gender                  -   Patient gender  (1=Female ; 0=Male)
polyuria                -   Whether the patient experienced excessive urination  (1=True ; 0=False)
polydipsia              -   Whether the patient experienced excessive thirst/excess drinking  (1=True ; 0=False)
sudden_weight_loss      -   Whether the patient had an episode of sudden weight loss  (1=True ; 0=False)
weakness                -   Whether the patient experienced episodes of feeling weak  (1=True ; 0=False)
polyphagia              -   Whether the patient had an episode of excessive/extreme hunger  (1=True ; 0=False)
genital_thrush          -   Whether the patient had a yeast infection or not  (1=True ; 0=False)
visual_blurring         -   Whether the patient experienced blurred vision  (1=True ; 0=False)
itching                 -   Whether the patient experienced episodes of itching  (1=True ; 0=False)
irritability            -   Whether the patient experienced episodes of irritability  (1=True ; 0=False)
delayed_healing         -   Whether the patient noticed delayed healing when wounded  (1=True ; 0=False)
partial_paresis         -   Whether the patient had an episode of weakening of muscle group(s)  (1=True ; 0=False)
muscle_stiffness        -   Whether the patient experienced any muscle stiffness  (1=True ; 0=False)
alopecia                -   Whether the patient experienced hair loss  (1=True ; 0=False)
obesity                 -   Whether the patient can be considered obese, based on BMI  (1=True ; 0=False)

class                   -   RESULT - Presence of Diabetes?    (1 = YES  ;  0 = NO)
"""


################################################


# Import data file into Pandas DataFrame
csv_data_file = "diabetes_data.csv"
df = pd.read_csv(csv_data_file)

# Feature selection (features - X)
X = df[[
    'age',
    'gender',
    'polyuria',
    'sudden_weight_loss',
    'weakness',
    'polyphagia',
    'genital_thrush',
    'visual_blurring',
    'itching',
    'irritability',
    'delayed_healing',
    'partial_paresis',
    'muscle_stiffness',
    'alopecia',
    'obesity'
]]  # 15 features

# Target selection (label - y)
y = df[['class']]


# SPLIT DATA INTO TRAINING & TESTING DATA
# NOTE: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


################################################


##################
# TRAINING MODEL #
##################

# Initialize Decision Tree Classifier
dt_clf = tree.DecisionTreeClassifier()

# Initialize the K-Nearest Neighbors (KNN) Classifier
knn_clf = KNeighborsClassifier()


# TRAIN Decision Tree Classifier
dt_clf = dt_clf.fit(X_train, y_train)

# TRAIN the K-Nearest Neighbors (KNN) Classifier
knn_clf = knn_clf.fit(X_train, y_train)


################################################


############################
# TESTING & ACCURACY SCORE #
############################


# {{{  DECISION TREE CLASSIFIER  }}} #

# Get test classifier (for accuracy score)
dt_test_prediction = dt_clf.predict(X_test)

# ACCURACY SCORE  -  Get prediction % for DT
prediction_percent_score_DT = accuracy_score(y_test, dt_test_prediction)
DT_accuracy_score_percentage = "{:.0%}".format(prediction_percent_score_DT)


########################


# {{{  KNN CLASSIFIER  }}} #

# Get test classifier (for accuracy score)
knn_test_prediction = knn_clf.predict(X_test)

# ACCURACY SCORE  -  Get prediction % for KNN
prediction_percent_score_KNN = accuracy_score(y_test, knn_test_prediction)
KNN_accuracy_score_percentage = "{:.0%}".format(prediction_percent_score_KNN)


################################################
################################################

##################
# CONSOLE OUTPUT #
##################


# Output feature list  -  a view of the data
print()
print("\n\t  *** Feature List: ***"
      "\n\t=========================")
print(X.head())
print(". " * 21)
print(X.tail())

print()
print("*" * 50)


# Output the full data shape  -  how many rows & columns there are
print("\n  [ Full Data Shape:", X.shape, "]")
print()
print("*" * 50)


input("\n Press <ENTER> to continue...")

print()
print("*" * 50)


########################


print()
print("\n\t  *** Accuracy Scores: ***"
      "\n\t============================")


# ACCURACY SCORE  -  Output prediction % for DT
print("\nDecision Tree Classifier prediction score: ", DT_accuracy_score_percentage)

# ACCURACY SCORE  -  Output prediction % for KNN
print("\nK-Nearest Neighbors Classifier prediction score: ", KNN_accuracy_score_percentage)


print("\n")
print("*" * 50)
