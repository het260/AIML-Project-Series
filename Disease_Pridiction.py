# Importing libraries
import numpy as np 
import pandas as pd 
from scipy.stats import mode 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix 

# Data Collection
# Define the path to the CSV file
DATA_PATH = "D://Nexus Info Internship/Phase - 2/Training.csv"

# Load the dataset, removing the last column since it's empty
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Data Preprocessing
# Calculate the number of instances for each disease in the dataset
disease_counts = data["prognosis"].value_counts()

# Create a temporary DataFrame to hold the disease names and their counts
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

# Set the size of the figure for better readability
plt.figure(figsize=(18, 8))

# Create a bar plot to visualize the distribution of diseases
sns.barplot(x="Disease", y="Counts", data=temp_df)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Display the plot
plt.title("Bar Plot to visualize the Distribution of Diseases")  # Set the title of the barplot
# plt.show()
plt.savefig("BarPlotofDistribution.png")

# Encoding the target value into numerical value using LabelEncoder 
# Initialize the LabelEncoder
encoder = LabelEncoder()

# Fit the LabelEncoder on the 'prognosis' column and transform it into numerical values
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Split the dataset into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Feature Selection and Data Preprocessing
# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing, and 80% for training
# random_state=24 ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Print the shape of the training and testing sets
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Cross Validation
# Defining scoring metric for k-fold cross validation 
def cv_scoring(estimator, X, y): 
	return accuracy_score(y, estimator.predict(X)) 

# Model Development and Cross Validation
# Initializing models to be evaluated
models = { 
	"SVC":SVC(), 
	"Gaussian NB":GaussianNB(), 
	"Random Forest":RandomForestClassifier(random_state=18) 
} 

# Producing cross validation score for the models 
for model_name in models: 
	model = models[model_name] 
	scores = cross_val_score(model, X, y, cv = 10, n_jobs = -1, scoring = cv_scoring) 
	
	# Print the results for each model
	print("=="*30) 
	print(model_name) 
	print(f"Scores: {scores}") 
	print(f"Mean Score: {np.mean(scores)}")

# Validation and Testing
# Training and testing SVM Classifier
svm_model = SVC()  # Initialize the Support Vector Machine classifier
svm_model.fit(X_train, y_train)  # Train the SVM classifier on the training data
preds = svm_model.predict(X_test)  # Predict the labels for the test data

# Print the accuracy on the training and test data for the SVM classifier
print(f"Accuracy on train data by SVM Classifier : {accuracy_score(y_train, svm_model.predict(X_train))*100}")
print(f"Accuracy on test data by SVM Classifier : {accuracy_score(y_test, preds)*100}")

# Create a confusion matrix for the SVM classifier on the test data
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))  # Set the figure size
sns.heatmap(cf_matrix, annot=True)  # Plot the heatmap of the confusion matrix with annotations
plt.title("Confusion Matrix for SVM Classifier on Test Data")  # Set the title of the heatmap
# plt.show()  # Display the heatmap
plt.savefig("CM_SVMClassifier.png")

# Validation and Testing
# Training and testing Naive Bayes Classifier
nb_model = GaussianNB()  # Initialize the Gaussian Naive Bayes classifier
nb_model.fit(X_train, y_train)  # Train the Naive Bayes classifier on the training data
preds = nb_model.predict(X_test)  # Predict the labels for the test data

# Print the accuracy on the training and test data for the Naive Bayes classifier
print(f"Accuracy on train data by Naive Bayes Classifier : {accuracy_score(y_train, nb_model.predict(X_train))*100}")
print(f"Accuracy on test data by Naive Bayes Classifier : {accuracy_score(y_test, preds)*100}")

# Create a confusion matrix for the Naive Bayes classifier on the test data
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))  # Set the figure size
sns.heatmap(cf_matrix, annot=True)  # Plot the heatmap of the confusion matrix with annotations
plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")  # Set the title of the heatmap
# plt.show()  # Display the heatmap
plt.savefig("CM_NaiveBayesClassifier.png")

# Validation and Testing
# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)  # Initialize the Random Forest classifier with a fixed random state
rf_model.fit(X_train, y_train)  # Train the Random Forest classifier on the training data
preds = rf_model.predict(X_test)  # Predict the labels for the test data

# Print the accuracy on the training and test data for the Random Forest classifier
print(f"Accuracy on train data by Random Forest Classifier : {accuracy_score(y_train, rf_model.predict(X_train))*100}")
print(f"Accuracy on test data by Random Forest Classifier : {accuracy_score(y_test, preds)*100}")

# Create a confusion matrix for the Random Forest classifier on the test data
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))  # Set the figure size
sns.heatmap(cf_matrix, annot=True)  # Plot the heatmap of the confusion matrix with annotations
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")  # Set the title of the heatmap
# plt.show()  # Display the heatmap
plt.savefig("CM_RandomForestClassifier.png")

# Model Development
# Training the models on whole data 
final_svm_model = SVC() 
final_nb_model = GaussianNB() 
final_rf_model = RandomForestClassifier(random_state=18) 

# Fit the models on the entire dataset
final_svm_model.fit(X, y) 
final_nb_model.fit(X, y) 
final_rf_model.fit(X, y) 

# Validation and Testing
# Reading the test data 
test_data = pd.read_csv("D://Nexus Info Internship/Phase - 2/Testing.csv").dropna(axis=1) 

# Splitting the test data into features and target
test_X = test_data.iloc[:, :-1] 
test_Y = encoder.transform(test_data.iloc[:, -1]) 

# Making prediction by take mode of predictions made by all the classifiers 
svm_preds = final_svm_model.predict(test_X) 
nb_preds = final_nb_model.predict(test_X) 
rf_preds = final_rf_model.predict(test_X) 

# Combining predictions by taking the mode of predictions made by all the classifiers
final_preds = [mode([i, j, k])[0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

# Calculating and printing the accuracy of the combined model on the test dataset
print(f"Accuracy on Test dataset by the combined model : {accuracy_score(test_Y, final_preds)*100}") 

# Generating and plotting the confusion matrix for the combined model
cf_matrix = confusion_matrix(test_Y, final_preds) 
plt.figure(figsize=(12,8)) 
sns.heatmap(cf_matrix, annot = True) 
plt.title("Confusion Matrix for Combined Model on Test Dataset") 
# plt.show()
plt.savefig("CM_CombinedModel.png")

# Web Application Development
# Import necessary libraries
import streamlit as st
from statistics import mode

# Extract symptoms from the features
symptoms = X.columns.values

# Creating a symptom index dictionary to encode the input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

# Creating a data dictionary to hold the symptom index and prediction classes
data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Defining the Function to predict disease
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    
    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        symptom = symptom.strip().title()  # Convert to title case for comparison
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1

    # Reshaping the input data and converting it into a suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # Generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # Making the final prediction by taking the mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

# Streamlit app
def main():
    st.title("Disease Prediction Model")
    st.write("Enter your symptoms separated by commas:")
    user_input = st.text_input("")
    if st.button("Predict"):
        result = predictDisease(user_input)
        st.write("Predictions:")
        st.write(f"RF Model Prediction: {result['rf_model_prediction']}")
        st.write(f"Naive Bayes Prediction: {result['naive_bayes_prediction']}")
        st.write(f"SVM Model Prediction: {result['svm_model_prediction']}")
        st.write(f"Final Prediction: {result['final_prediction']}")
        
        # Display the saved figures
        st.image("BarPlotofDistribution.png")
        st.image("CM_SVMClassifier.png")
        st.image("CM_NaiveBayesClassifier.png")
        st.image("CM_RandomForestClassifier.png")
        st.image("CM_CombinedModel.png")

if __name__ == "__main__":
    main()