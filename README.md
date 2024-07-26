# Sentiment-Analysis
Objective: Develop a model to automatically determine the sentiment of text data (e.g., positive, negative, or neutral) using a Naive Bayes classifier.

Description:
This project involves building a sentiment analysis system to classify text into different sentiment categories. We use the Naive Bayes algorithm due to its effectiveness in handling text data and its simplicity. The project includes:

Data Collection: Gather a dataset of text samples with known sentiment labels. Common sources include movie reviews, social media posts, or customer feedback. Here IMDB Dataset is used.

Data Preprocessing: Clean the text data by using Labelencoder and applying tokenization. Convert text into numerical features using Countvectorizer.

Model Training: Split the data into training and testing sets. Train a Naive Bayes classifier on the training set, which leverages Bayes' theorem and the assumption of feature independence.

Evaluation: Assess the model's performance using metrics such as accuracy, precision, recall, and F1-score on the test set.

Deployment: Implement the trained model to predict sentiment on new, unseen text data.

The Naive Bayes classifier is chosen for its efficiency and ease of implementation, making it well-suited for text classification tasks in sentiment analysis.

