import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer
import time

import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Start measuring runtime
start_time = time.time()

# Loading the Cleaned Dataset
data = pd.read_csv("C:/Users/kaush/OneDrive/Documents/Research paper/Customer Review Dataset/Restaurant_Review_cleaned_data.tsv", delimiter='\t', quoting=3)
corpus = []


# -->Data Preprocessing Step
for i in range(data.shape[0]):
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=data['Review'][i])
    review = review.lower()
    review_words = review.split()
    review_words = [word for word in review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review_words]
    review = ' '.join(review)
    corpus.append(review)

# Creating Bag of Words model using Count Vectorization
cv = CountVectorizer(ngram_range=(1, 2), max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

# Splitting Dataset into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=32)

# Initialize individual classifiers with best parameters
nb_classifier = MultinomialNB(alpha=0.9)
lr_classifier = LogisticRegression(C=1, random_state=0)

# Create a Voting Classifier with soft voting
voting_classifier = VotingClassifier(estimators=[
    ('multinomial_nb', nb_classifier),
    ('logistic_regression', lr_classifier)
], voting='hard')

# Fit the Voting Classifier to your training data
voting_classifier.fit(x_train, y_train)

# Predictions using the Voting Classifier
y_pred_voting = voting_classifier.predict(x_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred_voting)
precision = precision_score(y_test, y_pred_voting)
recall = recall_score(y_test, y_pred_voting)
f_score = f1_score(y_test, y_pred_voting)

print("\n***** Performance Metrics using Voting Classifier *****")
print("\nAccuracy: {}%".format(round(accuracy * 100, 2)))
print("Precision: {}".format(round(precision, 2)))
print("Recall: {}".format(round(recall, 2)))
print("F_Score: {}\n".format(round(f_score, 2)))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_voting)

# Plotting Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', annot_kws={"size": 25},xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

# End measuring runtime
end_time = time.time()

# Calculate the total runtime
runtime = end_time - start_time

# Print the runtime in seconds
print("Total runtime: {} seconds".format(round(runtime, 2)))


# Function to predict sentiment using the Voting Classifier
def predict_sentiment(sample_review):
    # Preprocess the input text as done before
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)

    # Transform the preprocessed text using CountVectorizer
    temp = cv.transform([final_review]).toarray()

    # Predict the sentiment using the Voting Classifier
    prediction = voting_classifier.predict(temp)

    # Return the sentiment prediction (1 for Positive, 0 for Negative)
    return prediction[0]

# Example text for prediction
sample_review = "The food is really good"

# Make a prediction
result = predict_sentiment(sample_review)

if result == 1:
    print("\n" + sample_review + " -> Positive Comment")
else:
    print("\n" + sample_review + " -> Negative Comment")
