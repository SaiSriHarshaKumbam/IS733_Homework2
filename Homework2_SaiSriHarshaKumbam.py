#!/usr/bin/env python
# coding: utf-8

# # Homework-2

# # Part-3 Wine Tasting Machine

# In[1]:


#Task-1 
#Read Red-wine.csv into python as a dataframe
import pandas as pd
file_path="C:/Users/harsha/Desktop/DataMining_733/Homework2/red_wine.csv"
# Read CSV file into DataFrame
df=pd.read_csv(file_path)
#Get the shape of the dataset
shape=df.shape
print("Shape of the dataset: ",shape)


# In[2]:



import pandas_profiling as pp

df.profile_report()


# In[3]:


#The below lines of code will create a profile report for the dataframe and will simply generate a report in the form of .html file 
from pandas_profiling import ProfileReport

# Create a profile report
profile = ProfileReport(df)

# Generate an HTML report
profile.to_file("red_wine_data_report.html")


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:



from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


# Encode the "type" column to be numeric (0 for "low" and 1 for "high")
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Prepare the data for modeling
X = df.drop('type', axis=1)  # Features
y = df['type']  # Target (0 for "low" and 1 for "high")

# Create instances of the classifiers
lr_classifier = LogisticRegression()
nb_classifier = GaussianNB()
svm_classifier = SVC(probability=True)  # Note: Setting probability=True for ROC AUC calculation
dt_classifier = DecisionTreeClassifier()
rf_classifier = RandomForestClassifier()

# Create a function to perform cross-validation and report performance metrics
def cross_val_and_report_performance(classifier, name):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    accuracy = cross_val_score(classifier, X, y, cv=kfold, scoring='accuracy')
    auc_scores = cross_val_score(classifier, X, y, cv=kfold, scoring='roc_auc')
    
    # Calculate baseline accuracy (majority class) and baseline AUC (random classifier)
    baseline_accuracy = np.max([np.mean(y == c) for c in y.unique()])
    baseline_auc = 0.5  # For a random classifier
    
    print(f"Model: {name}")
    print("Cross-Validation Accuracy: {:.2f} (+/- {:.2f})".format(accuracy.mean(), accuracy.std() * 2))
    print("Baseline Model Accuracy: {:.2f}".format(baseline_accuracy))
    print("Cross-Validation AUC: {:.2f} (+/- {:.2f})".format(auc_scores.mean(), auc_scores.std() * 2))
    print("Baseline Model AUC: {:.2f}".format(baseline_auc))
    print("------")

# Perform cross-validation and report performance for each model
cross_val_and_report_performance(lr_classifier, "Logistic Regression")
cross_val_and_report_performance(nb_classifier, "Naïve Bayes")
cross_val_and_report_performance(svm_classifier, "Support Vector Machine")
cross_val_and_report_performance(dt_classifier, "Decision Tree")
cross_val_and_report_performance(rf_classifier, "Random Forest")


# In[ ]:





# In[ ]:





# #Task-2

# In[9]:



from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from prettytable import PrettyTable


# Encode the "type" column to be numeric (0 for "low" and 1 for "high")
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Prepare the data for modeling
X = df.drop('type', axis=1)  # Features
y = df['type']  # Target (0 for "low" and 1 for "high")

# Create instances of the classifiers
lr_classifier = LogisticRegression()
nb_classifier = GaussianNB()
dt_classifier = DecisionTreeClassifier()
svm_linear_classifier = SVC(kernel='linear', probability=True)
svm_rbf_classifier = SVC(kernel='rbf', probability=True)
rf_classifier = RandomForestClassifier()

# Create a KFold cross-validation splitter with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define a function to calculate performance metrics (AUC and accuracy)
def calculate_metrics(classifier, X, y):
    if classifier is not None:
        auc_scores = cross_val_score(classifier, X, y, cv=kf, scoring='roc_auc')
        accuracy_scores = cross_val_score(classifier, X, y, cv=kf, scoring='accuracy')
        
        auc_mean = auc_scores.mean()
        accuracy_mean = accuracy_scores.mean()
    else:
        # For the baseline model
        auc_mean = 0.5  # Random classifier AUC
        accuracy_mean = np.max([np.mean(y == c) for c in y.unique()])  # Majority class accuracy
    return auc_mean, accuracy_mean

# Create a PrettyTable to format the output
table = PrettyTable()
table.field_names = ["Model", "AUC", "Accuracy"]

# Calculate AUC and accuracy for the baseline model
baseline_auc, baseline_accuracy = calculate_metrics(None, X, y)
table.add_row(["Baseline", f"{baseline_auc:.2f}", f"{baseline_accuracy:.2f}"])

# Calculate AUC and accuracy for Logistic Regression
lr_auc, lr_accuracy = calculate_metrics(lr_classifier, X, y)
table.add_row(["Logistic Regression", f"{lr_auc:.2f}", f"{lr_accuracy:.2f}"])

# Calculate AUC and accuracy for Naïve Bayes (GaussianNB)
nb_auc, nb_accuracy = calculate_metrics(nb_classifier, X, y)
table.add_row(["Naive Bayes", f"{nb_auc:.2f}", f"{nb_accuracy:.2f}"])

# Calculate AUC and accuracy for Decision Tree
dt_auc, dt_accuracy = calculate_metrics(dt_classifier, X, y)
table.add_row(["Decision Tree", f"{dt_auc:.2f}", f"{dt_accuracy:.2f}"])

# Calculate AUC and accuracy for SVM with linear kernel
svm_linear_auc, svm_linear_accuracy = calculate_metrics(svm_linear_classifier, X, y)
table.add_row(["SVM (Linear)", f"{svm_linear_auc:.2f}", f"{svm_linear_accuracy:.2f}"])

# Calculate AUC and accuracy for SVM with RBF kernel
svm_rbf_auc, svm_rbf_accuracy = calculate_metrics(svm_rbf_classifier, X, y)
table.add_row(["SVM (RBF)", f"{svm_rbf_auc:.2f}", f"{svm_rbf_accuracy:.2f}"])

# Calculate AUC and accuracy for Random Forest
rf_auc, rf_accuracy = calculate_metrics(rf_classifier, X, y)
table.add_row(["Random Forest", f"{rf_auc:.2f}", f"{rf_accuracy:.2f}"])

# Print the formatted table
print(table)


# #Question-3

# In[10]:



from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Fit the Random Forest model with the training data
rf_classifier.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_prob = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# #Question-4

# In[11]:


#The best model obtained from question-2 is Random Forest model that has the highest AUC value of 0.93 making it the best model


# In[ ]:





# In[21]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the white-wine.csv dataset into a Pandas DataFrame
white_wine_df = pd.read_csv("C:/Users/harsha/Desktop/DataMining_733/Homework2/white_wine.csv")

# Encode the "type" column to be numeric (0 for "low" and 1 for "high")
le = LabelEncoder()
white_wine_df['type'] = le.fit_transform(white_wine_df['type'])

# Prepare the data for modeling
X_white = white_wine_df.drop('type', axis=1)  # Features
y_white = white_wine_df['type']  # Target (0 for "low" and 1 for "high")

# Create the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)  # Set random seed for model

# Perform cross-validation and calculate AUC for white-wine data
auc_scores = cross_val_score(rf_classifier, X_white, y_white, cv=5, scoring='roc_auc')

# Calculate the mean AUC score
mean_auc = np.mean(auc_scores)

# Print the AUC score
print(f"Mean AUC for the Random Forest model on white-wine.csv: {mean_auc:.10f}")

# Comment on the performance
if mean_auc >= 0.9:
    print("The Random Forest model has excellent performance on the white-wine dataset.")
elif mean_auc >= 0.8:
    print("The Random Forest model has good performance on the white-wine dataset.")
else:
    print("The Random Forest model has moderate performance on the white-wine dataset.")


# In[ ]:





# In[ ]:




