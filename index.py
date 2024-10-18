#Import the libraries
import joblib
import numpy as np
import pandas as pd
import requests #for api call
import os
import datetime as dt #to extract date and time
import matplotlib.pyplot as plt #for plotting different graphs
import seaborn as sns #for visualization
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder #for label encoding

data_dir = r'C:/Users/abdelhamid.ma/PycharmProjects/pythonProject1/Maid.cc Mayar/Input Dataset/train - train.csv'
df = pd.read_csv(data_dir)

test_data_dir = r'C:/Users/abdelhamid.ma/PycharmProjects/pythonProject1/Maid.cc Mayar/Input Dataset/test - test.csv'
test_df = pd.read_csv(test_data_dir)
# print(df.head())

# **EDA Section:**
# basic EDA to understand the dataset
print(df.describe())
print(df.info())
print ("Number of rows:" )
print(df.shape[0])
print ("Number of columns: ")
print(df.shape[1])
print ("Null Values: ")


def print_unique_values_for_columns(df):
    columns = df.select_dtypes(include=['category', 'int64'])

    for column in columns:
        unique_values = df[column].unique()
        print(f"Unique values in column '{column}':\n{unique_values}\n")

print_unique_values_for_columns(df)

def reset_index_with_drop(df):
    return df.reset_index(drop=True)

df = reset_index_with_drop(df)
print(df.head())

# Data Cleaning

def remove_duplicates_and_count(dataframe):
    # Get the number of duplicates before removing
    num_duplicates_before = dataframe.duplicated().sum()
    print(f"Number of duplicate rows Before: {num_duplicates_before}")

    # Remove duplicate rows
    dataframe = dataframe.drop_duplicates()

    # Get the number of duplicates after removing
    num_duplicates_after = dataframe.duplicated().sum()
    print(f"Number of duplicate rows After: {num_duplicates_after}")

    return dataframe
df = remove_duplicates_and_count(df)
test_df = remove_duplicates_and_count(test_df)
### **4.   Handling Missing Data and Unclean Data as well:**

def calculate_null_summary(dataframe):
    # Calculate the sum of null entries for each column
    sum_null = dataframe.isnull().sum()

    # Calculate the percentage of null entries for each column
    perc_null = (dataframe.isnull().sum() / len(dataframe)) * 100

    return sum_null, perc_null

sum_null, perc_null = calculate_null_summary(df)
print(sum_null)
print(perc_null)

#since the no. of null values is less than 2% we can drop them but usually if the amount is larger we should invistigate a way to "fill" them
# Drop rows with any missing values
df_cleaned = df.dropna()
test_df_cleaned=test_df.dropna()
# Verify if any null values remain
print(df_cleaned.isnull().sum().sum())  # Should print 0

# Check the shape of the dataset before and after
print(f"Original shape: {df.shape}")
print(f"New shape after dropping rows: {df_cleaned.shape}")

sum_null, perc_null = calculate_null_summary(df_cleaned)
print(sum_null)
print(perc_null)

#----------------------------------------------------------------------------------#
analysis_df=df_cleaned
#EDA Questions
# 1. what outliers do we have
# analysis_df.boxplot(figsize=(15, 8))
# plt.title('Boxplot of Numerical Features')
# plt.show()

plt.figure(figsize = (20, 10))
x = 1

for column in analysis_df.columns :
    plt.subplot(7, 3, x)
    sns.boxplot(analysis_df[column])
    x+= 1

plt.tight_layout()
# plt.title('Boxplot of Numerical Features')
# plt.show()

#this should there are some outliers in the following columns "fc,px_height,three_g"
#we will invistage the number to see if further binning or scalling is needed

## getting the number of the outliers in fc column

analysis_df_description = analysis_df.describe()
fc_Q1 =  analysis_df_description['fc']['25%']
fc_Q3 = analysis_df_description['fc']['75%']
fc_IQR = fc_Q3 - fc_Q1

up_fence = fc_Q3 + (1.5 * fc_IQR)
lo_Fence = fc_Q1 - (1.5 * fc_IQR)

fc_outliers =  analysis_df[(analysis_df['fc'] < lo_Fence) | (analysis_df['fc']> up_fence)]
print(fc_outliers.count())

# there are 18 phone that has a front camera outlier value
def analyze_fc_histograms(dataframe):
    # Define the figure with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    # Histogram of the Front Camera megapixels (with outliers)
    dataframe["fc"].hist(bins=30, ax=ax[0], color='skyblue', edgecolor='black')
    ax[0].set_xlabel('Front Camera (Megapixels)')
    ax[0].set_ylabel('Count')
    ax[0].set_yscale('log')  # Optional: Set y-axis to log scale if distribution is skewed
    ax[0].set_title('Histogram of Front Camera Megapixels with Outliers')

    # Create a vector to contain `fc` values
    v = dataframe["fc"]

    # Exclude data points located beyond 3 standard deviations from the median (outliers)
    filtered_fc = v[~((v - v.median()).abs() > 3 * v.std())]

    # Plot the histogram of `fc` without outliers
    filtered_fc.hist(bins=30, ax=ax[1], color='lightcoral', edgecolor='black')
    ax[1].set_xlabel('Front Camera (Megapixels)')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Histogram of Front Camera Megapixels (Without Outliers)')

    # plt.show()


# Call the function with your DataFrame
analyze_fc_histograms(analysis_df)
#We filter out any values that are more than 3 standard deviations from the median, which removes extreme outliers.
#This gives a clearer picture of the central tendency and spread of the feature.

def analyze_ram_by_price_range(dataframe):
    # Aggregate mean and median RAM by price range
    table = dataframe.groupby('price_range')['ram'].agg(['mean', 'median']).reset_index()
    table.columns = ['Price_Range', 'Mean_RAM', 'Median_RAM']

    # Plotting the data for better visualization
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    table[['Mean_RAM', 'Median_RAM']].plot(ax=ax, kind='bar', color=['skyblue', 'coral'], edgecolor='black')
    ax.set_xlabel('Price Range')
    ax.set_ylabel('RAM (in MB)')
    ax.set_title('Mean and Median RAM by Price Range')
    ax.set_xticklabels(['Low Cost', 'Medium Cost', 'High Cost', 'Very High Cost'], rotation=0)

    # plt.show()

    # Print the aggregated table for reference
    print('Mean and Median RAM by Price Range\n')
    print(table.to_string(index=False))

# Call the function
analyze_ram_by_price_range(analysis_df)
#Higher price ranges are expected to have higher average and median RAM.

#relatively the dataset has very low number of outliers as well so it won't affect the ML Model performance
#--------------check correlation--------------

plt.figure(figsize = (20, 10))
sns.heatmap(analysis_df.corr(), annot = True)
# plt.show()

correlation = analysis_df.corr()
print(correlation['price_range'].sort_values(ascending = False)[1:])
#ram is the highest contributing attribute to price_range

#price range
sns.countplot(data=analysis_df, x='price_range')
plt.title('Distribution of Price Range')
plt.xlabel('Price Range')
plt.ylabel('Count')
# plt.show()
#there are 500 phones in each price_range

#we can start the building of the ML process


# Separate the features (X) and target (y)
x = analysis_df.drop(columns=['price_range'])  # Features
y = analysis_df['price_range']                 # Target

# Automatically detect all numerical columns (int and float types)
numerical_columns = x.select_dtypes(include=['int64', 'float64']).columns

# Initialize the scaler
scaler = StandardScaler()

# Scale only the numerical columns
x[numerical_columns] = scaler.fit_transform(x[numerical_columns])

# Display the first few rows of scaled feature variables
print("Scaled Feature Variables (X):")
print(x.head())

# Display the first few rows of the target variable
print("\nTarget Variable (y):")
print(y.head())

# **4. Classification Task:**
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model_names = ['Logistic Regression', 'Decision Tree','Random Forest','Support Vector','KNeighbors', 'XGBClassifier']
accuracies = []
precisions = []
recalls = []
f1_scores = []
conf_matrices = []
fprs= []
tprs= []
thresholds= []
aucs= []

#LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

logistic_reg = LogisticRegression(random_state=42, max_iter=1000)
logistic_reg.fit(x_train, y_train)

# Make predictions
y_pred = logistic_reg.predict(x_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Compute ROC AUC score for multiclass using One-vs-Rest (OvR)
y_prob = logistic_reg.predict_proba(x_test)  # Get probabilities
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

# Store results
accuracies.append(accuracy)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1)
conf_matrices.append(conf_matrix)
aucs.append(auc)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC (OvR):", auc)
print("\nClassification Report:\n", classification_rep)

#DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(x_train, y_train)

# Make predictions
y_pred = decision_tree.predict(x_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Compute ROC AUC score for multiclass using One-vs-Rest (OvR)
y_prob = logistic_reg.predict_proba(x_test)  # Get probabilities
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

# Store results
accuracies.append(accuracy)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1)
conf_matrices.append(conf_matrix)
aucs.append(auc)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC (OvR):", auc)
print("\nClassification Report:\n", classification_rep)

#randomforest
random_forest = RandomForestClassifier(random_state=42)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)
# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Compute ROC AUC score for multiclass using One-vs-Rest (OvR)
y_prob = logistic_reg.predict_proba(x_test)  # Get probabilities
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

# Store results
accuracies.append(accuracy)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1)
conf_matrices.append(conf_matrix)
aucs.append(auc)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC (OvR):", auc)
print("\nClassification Report:\n", classification_rep)


#SVM
svm_classifier = SVC(random_state=42)

svm_classifier.fit(x_train, y_train)

y_pred = svm_classifier.predict(x_test)
# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Compute ROC AUC score for multiclass using One-vs-Rest (OvR)
y_prob = logistic_reg.predict_proba(x_test)  # Get probabilities
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

# Store results
accuracies.append(accuracy)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1)
conf_matrices.append(conf_matrix)
aucs.append(auc)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC (OvR):", auc)
print("\nClassification Report:\n", classification_rep)

#KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

knn_classifier.fit(x_train, y_train)

y_pred = knn_classifier.predict(x_test)
# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Compute ROC AUC score for multiclass using One-vs-Rest (OvR)
y_prob = logistic_reg.predict_proba(x_test)  # Get probabilities
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

# Store results
accuracies.append(accuracy)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1)
conf_matrices.append(conf_matrix)
aucs.append(auc)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNeighbors Classifier Confusion Matrix')
plt.show()

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC (OvR):", auc)
print("\nClassification Report:\n", classification_rep)

#XGBClassifier Model:
xgb_classifier = XGBClassifier(random_state=42)

xgb_classifier.fit(x_train, y_train)

y_pred = xgb_classifier.predict(x_test)
# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Compute ROC AUC score for multiclass using One-vs-Rest (OvR)
y_prob = logistic_reg.predict_proba(x_test)  # Get probabilities
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

# Store results
accuracies.append(accuracy)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1)
conf_matrices.append(conf_matrix)
aucs.append(auc)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGB Classifier Confusion Matrix')
plt.show()

# Print metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC (OvR):", auc)
print("\nClassification Report:\n", classification_rep)

#Comparesion between the model performances on dataset:
model_comparison = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1 Score': f1_scores,
    # 'False Positive Rate' : fprs,
    # 'True Positive Rate' : tprs,
    # 'Thresholds': thresholds,
    'AUC': aucs
})

print(model_comparison)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.barh(model_comparison['Model'], model_comparison['Accuracy'], color='blue')
plt.xlabel('Accuracy')
plt.title('Accuracy Comparison')

plt.subplot(2, 2, 2)
plt.barh(model_comparison['Model'], model_comparison['Precision'], color='green')
plt.xlabel('Precision')
plt.title('Precision Comparison')

plt.subplot(2, 2, 3)
plt.barh(model_comparison['Model'], model_comparison['Recall'], color='orange')
plt.xlabel('Recall')
plt.title('Recall Comparison')

plt.subplot(2, 2, 4)
plt.barh(model_comparison['Model'], model_comparison['F1 Score'], color='red')
plt.xlabel('F1 Score')
plt.title('F1 Score Comparison')

plt.tight_layout()
plt.show()

max_accuracy = model_comparison['Accuracy'].max()
max_precision = model_comparison['Precision'].max()
max_recall = model_comparison['Recall'].max()
max_f1_score = model_comparison['F1 Score'].max()

print("Maximum Accuracy:", max_accuracy)
print("Maximum Precision:", max_precision)
print("Maximum Recall:", max_recall)
print("Maximum F1 Score:", max_f1_score)


#from comparing the different model performance I would choose logistic regression model to predict our price category
#given that the accuracies of the model is at 95% no need to optimize it further given that the use case is not critical and can allow a small margin of error
# Save the trained model to a file
joblib.dump(logistic_reg, 'model.pkl')
print("Model saved as model.pkl")
