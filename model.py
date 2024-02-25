import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

file_path = 'dataFinal.csv'
df = pd.read_csv(file_path)
# print(df)

x = df['text']
# print(x)
y = df['Pattern Category']
# print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(X_test)
# print(y_test)
# print(y_train)


# Convert text data into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# print(X_train_tfidf)

# Create and train the SVM model
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train_tfidf, y_train)


# ...........................

# Make predictions on the test set
y_pred = svm_model.predict(X_test_tfidf)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Create a pie chart to visualize the distribution of classes
plt.figure(figsize=(12, 10))  # Increase the figure size
class_counts = pd.Series(y_test).value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution of Classes')
plt.legend(title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))  # Add a legend outside the pie chart

# Add a box with labels and percentages
box_text = '\n'.join(f'{label}: {percentage:.1f}%' for label, percentage in zip(class_counts.index, class_counts / class_counts.sum() * 100))
plt.text(1.2, 0.5, box_text, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=12, verticalalignment='center')

plt.show()

