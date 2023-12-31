import matplotlib.pyplot as plt
import numpy as np


# Sample data for sentiment distribution (update as per your dataset)
sentiment = np.array([714, 710])  # Negative and Positive counts

# Create a figure and axis
fig, ax = plt.subplots()

index = [0, 1]

# Create separate bar plots for "Negative" and "Positive" categories
ax.bar(index[0], sentiment[0], color="red")
ax.bar(index[1], sentiment[1], color="green")
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.title('Sentiment Distribution', fontsize=12)
ax.set_xticks(index)
ax.set_xticklabels(['Negative', 'Positive'], fontsize=12)

# Show the plot
plt.show()

'''
models = ['DT', 'NB', 'RF', 'SVM', 'LR', 'KNN']
accuracies = [72.26, 84.91, 77.86, 79.08, 63.02, 64.23]

# Unique colors for the bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# Create the bar plot
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)
plt.xlabel('Models', fontsize=15)
plt.ylabel('Accuracy (%)', fontsize=15)
plt.title('Accuracy of Models Before Hyper-parameter Tuning', fontsize=15)
plt.ylim(60, 90)

# Annotate each bar with its value
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.annotate(f'{height:.2f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 7),  # 7 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=14, fontweight='bold', color="black")
    
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
'''

'''
models = ['DT', 'NB', 'RF', 'SVM', 'LR', 'KNN']
accuracies = [73.88, 84.91, 79.08, 81.02, 81.02, 70.8]

# Unique colors for the bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Create the bar plot
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=colors)
plt.xlabel('Models', fontsize=15)
plt.ylabel('Accuracy (%)', fontsize=15)
plt.title('Accuracy of Models After Hyper-parameter Tuning', fontsize=15)
plt.ylim(60, 90)

# Annotate each bar with its value
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.annotate(f'{height:.2f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 7),  # 7 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=14, fontweight='bold', color="black")

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
'''