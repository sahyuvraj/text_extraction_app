import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import pytesseract
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import string
import re
import seaborn as sns

import string

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def remove_pun(text):
    for pun in string.punctuation:
        text = text.replace(pun, "")
    text = text.lower()
    return text


def language_indentification(txt):
    df = pd.read_csv('Language Detection.csv')
    df['Text'] = df['Text'].apply(remove_pun)

    X = df.iloc[:, 0]
    Y = df.iloc[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

    vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), analyzer='char')

    model_pipe2 = pipeline.Pipeline([('vec', vec), ('clf', SVC())])
    model_pipe2.fit(x_train, y_train)
    y_predict2 = model_pipe2.predict([txt])
    return y_predict2

def extract(img):
    text = pytesseract.image_to_string(img)
    language = language_indentification(text)
    return text, language

# if __name__ == "__main__":
#     img = cv2.imread("C:/Users/hp/Desktop/jupyter/test images/letter-1.png")
#     txt, lang = extract(img)
#     print(txt)
#     print(lang)
