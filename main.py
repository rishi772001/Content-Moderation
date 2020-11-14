from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


# import local modules
from src.naive_bayes import *
from src.predict import *
from src.svm import *


# from google.colab import drive
# drive.mount('/content/drive')


naivemodelcount, naive_count_vect = joblib.load("trained models/naive_count.sav")

traindata, testdata, trainlabel, testlabel = read()

print(predict_line(naivemodelcount, input("Enter text: "), naive_count_vect))

# print("multinomial naive bayes accuracy(count vectorization) = ",predict_accuracy(naivemodelcount, testdata, testlabel, naive_count_vect))
