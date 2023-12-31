import matplotlib.pyplot as plt

# Data
models = ['DT', 'NB', 'RF', 'SVM', 'LR', 'KNN', 'C1', 'C2', 'C3', 'C4']
time = [5.44, 4.09, 10.07, 7.02, 6.26, 7.35, 4.17, 4.05, 3.57, 3.94]

# Unique colors for the bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create the bar graph with bars filled with unique colors
plt.figure(figsize=(16, 12))
bars = plt.bar(models, time, color=colors)

plt.xlabel('Models', fontsize=30)
plt.ylabel('Time', fontsize=30)
plt.title('Processing Time of Models', fontsize=30)
plt.yticks(fontsize=30)
plt.xticks(rotation=45, fontsize=30)

# Annotate each bar with its value
for bar, Time in zip(bars, time):
    height = bar.get_height()
    plt.annotate(f'{Time:.2f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 7),  # Adjust vertical offset
                 textcoords="offset points",
                 ha='center',
                 fontsize=20,
                 fontweight='bold',
                 color='black')

plt.tight_layout()
plt.show()
