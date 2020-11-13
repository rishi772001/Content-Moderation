from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
# import matplotlib.pyplot as plt
import pandas as pd
# import numpy as np
# import itertools

# from google.colab import drive
# drive.mount('/content/drive')


def analyze(text):
    return "".join([i.strip() for i in text if i not in "!#$%&'()*+, -./:;<=>?@[\]^_`{|}~"])

count_vectorizer = CountVectorizer(stop_words='english')

def train_model(train_data, train_label):
    # Feature Extraction
    count_train = count_vectorizer.fit_transform(train_data)
    modelNB = MultinomialNB()
    modelNB.fit(count_train, train_label)
    return modelNB

def predict_line(model, inp):
    inp = list(inp)
    count_input = count_vectorizer.transform(inp)
    prediction = model.predict(count_input)
    if prediction[0] == 0:
        return True
    else:
        return False

def predict_accuracy(model, test, test_label):
    count_test = count_vectorizer.transform(test)
    prediction = model.predict(count_test)
    accuracy_score = metrics.accuracy_score(test_label, prediction)
    return accuracy_score

input_val = ["you are an idiot"]
df=pd.read_csv('comments_dataset.csv')

label=df["Insult"]
df=df.drop('Insult',axis=1)

traindata,testdata,trainlabel,testlabel = train_test_split(df['Comment'],label,test_size=0.20,random_state=53)

model = train_model(traindata, trainlabel)
# print(predict_line(model, input("Enter text")))
print(predict_accuracy(model, testdata, testlabel))
