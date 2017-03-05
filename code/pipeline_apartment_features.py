#Our feature construction class will inherit from these two base classes of sklearn.
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

class ApartmentFeaturesVectorizer(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, num_features = 400):
        self.num_features = num_features
        
    def fit(self, X,y):
        self.tfidf = CountVectorizer(stop_words='english',\
            max_features=self.num_features)
        self.tfidf.fit(X['proc_features'])
        return self
        
    def transform(self, X):
        X_sparse = self.tfidf.transform(X['proc_features'])
        del X['proc_features']
        X = sparse.hstack([X, X_sparse]).tocsr()
        return X