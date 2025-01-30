import numpy as np
import pandas as pd
import string


# Loading the dataset to train the model
df = pd.read_csv('spam.csv', encoding='latin-1')

# print(df.shape) just for checking the structure of the dataset

# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True) #since it doesn't contain any relevant information


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()  # to label SPAM as 1 and NOT-SPAM as 0

df['Target'] = encoder.fit_transform(df['Target'])

# print(df.sample(5)) just for debugging purpose


# missing values
# print(df.isnull().sum()) for checking null values

# check for duplicate values
# print(df.duplicated().sum()) for checking duplicate values

# remove duplicates
df = df.drop_duplicates(keep='first')

# print(df.duplicated().sum()) for checking duplicate values

# print(df.shape) just for debugging purpose

# Just to check the number of SPAM and NOT-SPAM msgs in the dataset
print(df['Target'].value_counts())

import matplotlib.pyplot as plt
plt.pie(df['Target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# Data Preprocessing

import nltk
nltk.download('punkt')

# num of characters
df['Num_Of_Characters'] = df['Text'].apply(len)


# num of words
nltk.download('punkt_tab')
df['Num_Of_Words'] = df['Text'].apply(lambda x:len(nltk.word_tokenize(x)))

# num of sentences
df['Num_Of_Sentences'] = df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)))


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()

nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))      
    return " ".join(y)

df['Transformed_text'] = df['Text'].apply(transform_text)

# print(df.sample(5)) just for debugging purpose
# print(transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."))

# Model Building

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)


X = tfidf.fit_transform(df['Transformed_text']).toarray()
y = df['Target'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


mnb = MultinomialNB()

mnb.fit(X_train,y_train)

y_pred = mnb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f} %")
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print(f"Precision: {precision_score(y_test,y_pred) * 100:.2f} %")

import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))