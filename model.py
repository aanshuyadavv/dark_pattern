import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

file_path = 'dataFinal.csv'
df = pd.read_csv(file_path)
# print(df)

x = df['text']
# print(x)
y = df['Pattern Category']
# print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
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

# Select the index you want to predict
selected_index = 0  # Change this to the desired index

myLine = 'Really good quality contacts and they work well on my eyes.Would definitely order again!!!!'

# Extract the text and true label for the selected index
selected_text = X_test.iloc[selected_index]
# print(selected_text)
true_label = y_test.iloc[selected_index]
# print(true_label)

# Convert the selected text into numerical features using TF-IDF
# selected_text_tfidf = tfidf_vectorizer.transform([selected_text])
selected_text_tfidf = tfidf_vectorizer.transform([myLine])

# Make a prediction
prediction = svm_model.predict(selected_text_tfidf)[0]
if((prediction!='Not Dark Pattern')):
    print('Dark Pattern Detected, Type Of Dark Pattern Is:', prediction)
else:
    print(prediction)








































# ...........................

# print(X_test_tfidf)
# Make predictions on the test set
# y_pred = svm_model.predict(X_test_tfidf)
# y_pred = svm_model.predict(X_test_tfidf[2])
# print(X_test_tfidf)
# print(y_pred)
# print(y_test)


# Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)

# Display the results
# print("\nAccuracy:", accuracy)
