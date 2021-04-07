import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


df = pd.read_csv("Eluvio_DS_Challenge.csv")
dfdata = df.drop(['time_created','date_created','down_votes','over_18','author','category'],axis='columns')

#Test train split - 90/10 split
X_train, X_test, y_train, y_test = train_test_split(dfdata.title, dfdata.up_votes, test_size=0.10)

#implementing countvectorizer on training data for multinomial bayes
v = CountVectorizer(analyzer=lambda x: x)
X_t = v.fit_transform(X_train).toarray()

#multinomal Bayes Model 
model = MultinomialNB()
model.fit(X_t, y_train)
X_tt = v.transform(X_test)
y_pred = model.predict(X_tt)

#K-Fold Cross Validation
print("N-fold Cross validation scores:", cross_val_score(model, X_t,y_train,cv=8))
print("Average accuracy score of this model is:" ,np.mean(cross_val_score(model, X_t,y_train,cv=8)))

#Final model scores and confusion matrix
print("Prediction accuracy Score for test data is:" ,model.score(X_tt, y_test)*100,"%")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

