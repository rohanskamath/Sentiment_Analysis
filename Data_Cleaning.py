import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/kaush/OneDrive/Documents/Research paper/Customer Review Dataset/Restaurant_Reviews.tsv", delimiter='\t', quoting=3)
'''
                                                      ***** 1. Data Cleaning Step *****
                                                      
1) Droping null and empty columns using ( data.drop(columns=['Name of the column',..],inplace=True) )
2) To rename the columns if valid column names are  not defined then, data.rename(columns={'column1':'rename'},inplace=True)
3) To see Sample data use, data.sample(5)
4) If column contains Binary value Text Eg:- "Positive-0" or "Negative-1" ->
        from sklearn.preprocessing import LabelEncoder
        encoder=LabelEncoder()
        data['liked']=encoder.fit_transform(data['liked'])
5) For Missing values:-
        data.isnull().sum()
6) Check for duplicate values

'''
print(data.shape)
sum_dups = "\nNo. of Duplicates before removal: " + str(data.duplicated().sum())
print(sum_dups)
data1 = data.drop_duplicates()
print("Shape of DataFrame after removing duplicates:", data1.shape)
sum_dups_after_removal = "\nNo. of Duplicates after removal: " + str(data1.duplicated().sum())
print(sum_dups_after_removal)

data1.to_csv('C:/Users/kaush/OneDrive/Documents/Research paper/Customer Review Dataset/Restaurant_Review_cleaned_data.tsv', sep='\t', index=False)

# Data Visualization using Bar Graph
sentiment_counts = data1['Liked'].value_counts()
print(sentiment_counts)

plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Distribution')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.show()

'''
                                                    ***** 2. Data Preprocessing Step *****
1) Removing Special characters
2) Converting Entire review to Lower case
3) Tokenizing the review by words
4) Removing the stop words
5) Stemming the words
6) Feature Extraction:- Bow Method (Bag-of-Words) or "TF-IDF Method"

'''
'''
                                                    ***** 3. Model Building Step *****
1) Spliting Dataset to training and test data
2) Fitting Naive Bayes,SVM,Random Forest,Decision Trees to the Training Set
3) Preditcing test data result
4) Plotting Confusion matrix
5) Performance matrix (Calculating the Accuracy,Precision,Recall,F-score )

'''

'''
                                                    ***** 3. Model Building Step *****
                                                    
                                            Predicting Values using created Classification Models
'''

'''
                                                    ***** 4. Results representing in Bar Graph ****

'''




'''
models = ['Decision Trees', 'Naive Bayes', 'Random Forest', 'SVM', 'Logistic Regression', 'KNN']
accuracies = [73.88, 84.91, 79.08, 81.02, 81.02, 70.8]

# Create the bar plot
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red','yellow','brown'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Models After Hyper-parameter Tuning')
plt.ylim(65, 90)

# Annotate each bar with its value, adjusting vertical offset and font size
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}%', 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 7),  # Adjust vertical offset here
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color="black")

plt.show()
'''