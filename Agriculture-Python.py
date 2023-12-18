#!/usr/bin/env python
# coding: utf-8

# # Smart Fields, Rich Yields: A Tech-Driven Harvest Quest

# <em>Problem description</em>: In the agricultural industry, characterized by intense competition, there exists a critical deficiency in comprehensive indicators designed to enhance the efficiency and outcomes of the crop lifecycle, encompassing the stages of planting, growing, and harvesting. The absence of robust metrics and benchmarks hinders farmers and stakeholders from making informed decisions, thereby limiting the industry's potential for optimization and sustainable growth. This lack of indicators not only jeopardizes individual farm productivity but also poses a broader challenge to the industry's ability to adapt to evolving demands and advancements in agricultural practices. Addressing this deficiency is paramount for fostering innovation, improving resource management, and ensuring the long-term viability of the agricultural sector.

# <em>How the solution will be used</em>: 
# 
# **Model Development:**
# 
# <em>Data Collection</em>: Gather relevant data pertaining to the crop lifecycle, including planting, growing, and harvesting phases. This dataset should encompass diverse variables such as weather conditions, soil quality, irrigation practices, and historical crop performance.
# 
# <em>Feature Engineering</em>: Identify and preprocess key features within the dataset that significantly influence the crop lifecycle.
# 
# <em>Model Training</em>: Utilize machine learning algorithms to develop a predictive model capable of analyzing the dataset and making accurate predictions related to crop outcomes.
# 
# <em>Evaluation and Tuning</em>: Assess the model's performance using validation datasets, and fine-tune parameters to optimize predictive accuracy and reliability.

# ## Import Dataset

# In[89]:


import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

import seaborn as sns


# Import the data frame

# In[9]:


blockOfData = pd.read_csv('Crop_recommendation.csv')


# Inspect the data frame

# In[20]:


print(blockOfData.shape)
blockOfData.head(1000)


# In[17]:


blockOfData['label'].unique()


# In[22]:


blockOfData.dtypes


# ## EDA

# Reference data frame information

# In[23]:


blockOfData.info()


# Inspect any null values

# In[41]:


blockOfData.isna().any()


# Search for duplicates

# In[46]:


blockOfData.duplicated().value_counts()


# Clean data

# In[107]:


strings =list(blockOfData.dtypes[blockOfData.dtypes=='object'].index)
strings


# In[108]:


for col in strings:
    blockOfData[col] = blockOfData[col].str.lower().str.replace(' ','_')


# In[109]:


blockOfData.head()


# Horizontal BC

# In[38]:


# Group and aggregate temperature data
data_by_temp = blockOfData.groupby('label')['temperature'].mean().sort_values()

# Define a color for the bars (e.g., 'skyblue')
color_of_bar = 'skyblue'

# Plot the data with the specified color
fig, ax = plt.subplots(figsize=(15, 5))
data_by_temp.plot(kind='barh', ax=ax, color=bar_color)

# Set axis labels and title
ax.set_xlabel('Temp (Celsius)')
ax.set_ylabel('Crop Name')
ax.set_title('How Much Humidity Does Each Crop Require?')

# Display the plot
plt.show()


# Descriptive Analysis

# In[47]:


blockOfData.describe(percentiles=[.0, .25, .5, .75, .9, .95, .99, .1]).T


# Calculate the median

# In[48]:


np.median(blockOfData['rainfall'])


# Examine each label's values

# In[45]:


# Apply the lowercase method to address potential matching concerns.

blockOfData["label"] = blockOfData["label"].str.lower()

# Create a subset dataframe specifically for "coffee" labels
coffee_df = blockOfData[blockOfData["label"] == "coffee"]

# Display the dataframe containing only "coffee" labels
print(coffee_df)


# Histogram

# In[67]:


feat = plt.figure(figsize=(8, 10))
ax = plt.gca()
blockOfData.hist(bins=50, ax=ax, layout=(4, 2), column=["rainfall", "temperature", "N", "P"])

plt.tight_layout()
plt.show()           


# In[69]:


print(blockOfData.head())
print(blockOfData.columns)
text = blockOfData['K'].iloc[0]


# In[93]:


# Increase the size of the heatmap
plt.figure(figsize=(12, 10))

# Create a heatmap of the correlation matrix for the DataFrame 'df'
correlation_matrix = blockOfData.corr()

# Adjust font size and rotate axis labels
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)

# Display the heatmap
plt.show()


# ## Model Training

# In[111]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report


# Separate the features and target labels

# In[123]:


features = blockOfData[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = blockOfData['label']
labels = blockOfData['label']


# In[124]:


# Initialize empty lists to store model names and corresponding accuracies
model_list = []
accuracy_list = []


# Split the test and train data

# In[125]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# Decision Tree Model

# In[128]:


# Initialize the Decision Tree Classifier with specified parameters
decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)

# Train the Decision Tree model using the training data
decision_tree.fit(Xtrain, Ytrain)

# Make predictions on the test data
predicted_values = decision_tree.predict(Xtest)

# Calculate and store the accuracy
accuracy = metrics.accuracy_score(Ytest, predicted_values)
accuracy_list.append(accuracy)

# Append the model name to the list
model_list.append('Decision Tree')

# Display the accuracy and classification report
print("Decision Tree's Accuracy is: {:.2f}%".format(accuracy * 100))
print(classification_report(Ytest, predicted_values))


# Cross-validation score

# In[133]:


from sklearn.model_selection import cross_val_score

cross_val_scores = cross_val_score(decision_tree, features, target, cv=5)


# Tune the decision tree parameters

# In[136]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


# In[138]:


# Assume 'features' and 'target' is the feature matrix and target variable
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define the Decision Tree model
decision_tree = DecisionTreeClassifier()

# Define the hyperparameters and their possible values to search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# Logistic Regression Model

# In[148]:


from sklearn.linear_model import LogisticRegression


# In[152]:


# Assuming you have 'test_X' and 'test_y' for your test set
predictions = logistic_regression_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy of the Logistic Regression model
print(f"Logistic Regression (accuracy): {accuracy * 100}%")


# Tune the LRM parameters

# In[153]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[155]:


# Assume 'x_train' and 'y_train' are in the training data
# Create a Logistic Regression model
logistic_regression_model = LogisticRegression()

# Define the hyperparameters and their possible values to search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],  # Regularization type
    'solver': ['liblinear', 'lbfgs']  # Optimization algorithm
}

# Create the GridSearchCV object
grid_search = GridSearchCV(logistic_regression_model, param_grid, cv=5, scoring='accuracy')

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best Logistic Regression model
best_logistic_regression_model = grid_search.best_estimator_

# Make predictions on the test set
predictions = best_logistic_regression_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Tuned Logistic Regression (accuracy): {:.2f}%".format(accuracy * 100))


# ### Further Development
# 
# **Web Application Development**:
# 
# <em>User Interface Design</em>: Create an intuitive and visually appealing web interface accessible to farmers. This interface should be designed for ease of use and understanding.
# 
# <em>Integration with Model</em>: Incorporate the trained machine learning model into the web application, allowing users to input relevant data and receive predictions regarding their crop lifecycle.
# 
# <em>Interactivity</em>: Implement features that enable users to interact with the model, such as adjusting input parameters, visualizing predictions, and receiving recommendations.
# 
# <em>User Feedback Mechanism</em>: Integrate a feedback system to gather user input and continuously improve the model's accuracy and relevance.
# 
# <em>Accessibility and Scalability</em>: Ensure the web application is easily accessible across different devices and scalable to accommodate potential future enhancements or increased user traffic.

# **Deployment and Maintenance**:
# 
# <em>Cloud Deployment</em>: Deploy the machine learning model and the web application on a cloud platform for scalability and accessibility.
# 
# <em>Monitoring and Updates</em>: Implement monitoring tools to track the performance of the model and application. Regularly update the model with new data to enhance its predictive capabilities.
# 
# <em>User Support</em>: Provide ongoing support and resources to assist farmers in effectively utilizing the web application and understanding the insights provided by the machine learning model.

# 
