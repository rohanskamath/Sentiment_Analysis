import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer

import pandas as pd
import numpy as np 

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Accuracy, Precision , Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Confusion Matrix
from sklearn.metrics import confusion_matrix

# Plotting the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns 

import time
# Start measuring runtime
start_time = time.time()

# Loading the Cleaned Dataset
data = pd.read_csv("C:/Users/kaush/OneDrive/Documents/Research paper/Customer Review Dataset/Restaurant_Review_cleaned_data.tsv", delimiter='\t', quoting=3)
data.shape
data.head()
corpus=[]

# -->Data Preprocessing Step
for i in range (data.shape[0]):
    review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=data['Review'][i])
    review=review.lower()
    review_words=review.split()
    review_words=[word for word in review_words if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review_words]
    review=' '.join(review)
    corpus.append(review)

print("\n***Sample 10 Preprocessed data from dataset***\n")
print(corpus[0:10])

# Creating Bag of Words model using Count Vectorization
cv=CountVectorizer(ngram_range=(1, 2), max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=data.iloc[:, 1].values


# -->Model Building Step

# Spliting Dataset to training and test data
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=32)

# Fitting Naive Bayes to the Training Set
classifier = RandomForestClassifier(n_estimators=100, random_state=32)
classifier.fit(x_train,y_train)

# Preditcing test data result
y_pred=classifier.predict(x_test)


# Calculating the Accuracy,Precision,Recall,F-score 
accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f_score=f1_score(y_test, y_pred)

print("\n***** Performance Matrices *****")
print("\nAccuracy: {}%".format(round(accuracy*100,2)))
print("Precision: {}".format(round(precision,2)))
print("Recall: {}".format(round(recall,2)))
print("F_Score: {}\n".format(round(f_score,2)))

cm=confusion_matrix(y_test,y_pred)

# Plotting Confusion Matrix
plt.figure(figsize=(10, 6))
ax=sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', annot_kws={"size": 25},xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=20)
plt.xlabel('Predicted Values',fontsize="18")
plt.ylabel('Actual Values',fontsize="18")
plt.show()

# Hyperparameter tuning the Naive Bayes Classifier
best_accuracy=0.0
n_estimator=0.0
for i in np.arange(50, 201, 50):
        temp_classifier = RandomForestClassifier(n_estimators=i, random_state=0)
        temp_classifier.fit(x_train, y_train)
        temp_y_pred=temp_classifier.predict(x_test)
        score=accuracy_score(y_test,temp_y_pred)
        print("Accuracy score for n_estimators {} is: {}%".format(round(i,1),round(score*100,2)))
        if score>best_accuracy:
            best_accuracy=score
            n_estimator=i
print("\nThe Best Accuracy is {} with n_estimators value as {}".format(round(best_accuracy*100,2),round(n_estimator,1)))


classifier = RandomForestClassifier(n_estimators=n_estimator, random_state=0)
classifier.fit(x_train, y_train)

# End measuring runtime
end_time = time.time()

# Calculate the total runtime
runtime = end_time - start_time

# Print the runtime in seconds
print("Total runtime: {} seconds".format(round(runtime, 2)))

# Prediction
def predict_sentiment(sample_review):
    sample_review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sample_review)
    sample_review=sample_review.lower()
    sample_review_words=sample_review.split() 
    sample_review_words=[word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    final_review=[ps.stem(word) for word in sample_review_words]
    final_review=' '.join(final_review)
    
    temp=cv.transform([final_review]).toarray()
    return classifier.predict(temp)
    
    
sample_review="The food is really bad"

if predict_sentiment(sample_review):
    print("\n"+sample_review+"->Positive Comment")
else:
    print("\n"+sample_review+"->Negative Comment")