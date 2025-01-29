import streamlit as st 
import pickle
import re
import nltk
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")

ps_stem = PorterStemmer()
lemma = WordNetLemmatizer()

def text_transform(text):
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text = [lemma.lemmatize(word) for word in text.split() if word not in stopwords.words('english')]
    text= " ".join(text)
            
    return text


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter a input: ")



if st.button("Predict"):
    # 1.preprocess
    processed_sms = text_transform(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([processed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    print(result)
    # 4. display 
    if result ==1:
        st.header("Spam")
    else:
        st.header("Not Spam")




