# SMS-Spam-Detection
Detecting Spam SMS's messages using machine learning algorithms.Utilizing NLP techniques such as tf-idf vectorization and machine learning algorithms like naive bayes and logistic regression for classifications. Integrated with Streamlit  to create an interactive web app where users can input a message and instantly check if it's spam or not.

## What I did Why did
This project focuses on detecting spam SMS messages using machine learning. The text is preprocessed using Lemmatization for better accuracy, and the TF-IDF vectorizer is applied to extract meaningful features. The dataset is trained using Multinomial Naïve Bayes and Logistic Regression.

The Multinomial Naïve Bayes model had a zero False Positive Rate, but a high False Negative Rate, meaning it missed a lot of spam messages. On the other hand, Logistic Regression had a zero False Negative Rate but a higher False Positive Rate. Since I believe that missing a spam message (classifying it as ham) is worse than marking a legitimate message as spam, I chose Logistic Regression as the final model.

The project is deployed using Streamlit, allowing users to enter a message and instantly check if it's spam or not.


