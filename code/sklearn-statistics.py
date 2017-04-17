from __future__ import print_function
import os
import sys
import operator
import math
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
from scipy import sparse
from collections import defaultdict
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import  Ridge, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from kernel_mlpclf import mlp_features

np.random.seed(42)

class CategoricalTransformer():
    def __init__(self, column_name, k=5.0, f=1.0, r_k=0.01, folds=5):
        self.k = k
        self.f = f
        self.r_k = r_k
        self.column_name = column_name
        self.folds = folds

    def _reset_fold(self):
        if hasattr(self, '_one_fold_mapping'):
            self._one_fold_mapping = {}
            self.glob_med = 0
            self.glob_high = 0

    def _fit_one_fold(self, X):
        self._reset_fold()

        tmp = X.groupby([self.column_name, 'interest_level']).size().\
            unstack().reset_index()
        tmp = tmp.fillna(0)

        tmp['record_count'] = tmp['high'] + tmp['medium'] + tmp['low']
        tmp['high_share'] = tmp['high']/tmp['record_count']
        tmp['med_share'] = tmp['medium']/tmp['record_count']

        self.glob_high = tmp['high'].sum()/tmp['record_count'].sum()
        self.glob_med = tmp['medium'].sum()/tmp['record_count'].sum()

        # Get weight function
        tmp['lambda'] = 1.0/(1.0+np.exp(np.float32(tmp['record_count']-self.k).\
            clip(-self.k,self.k)/self.f))
        
        # Blending
        tmp['w_high_'+self.column_name] =\
            (1.0-tmp['lambda'])*tmp['high_share']+tmp['lambda']*self.glob_high
        tmp['w_med_'+self.column_name] =\
            (1.0-tmp['lambda'])*tmp['med_share']+tmp['lambda']*self.glob_med

        # Adding random noise
        tmp['w_high_' + self.column_name] = tmp['w_high_' + self.column_name]*\
            (1+self.r_k*(np.random.uniform(size = len(tmp))-0.5))
        tmp['w_med_' + self.column_name] = tmp['w_med_' + self.column_name]*\
            (1+self.r_k*(np.random.uniform(size = len(tmp))-0.5))

        self._one_fold_mapping = tmp[['w_high_' + self.column_name,\
            'w_med_' + self.column_name,  self.column_name]]
        return self

    def _transform_one_fold(self, X):
        X = pd.merge(X,self._one_fold_mapping,how='left',on=self.column_name)
        return X[['w_high_' + self.column_name,'w_med_' + self.column_name]]

    def fit_transform_train(self, X, y):
        kfold = StratifiedKFold(self.folds)
        res = np.ones((X.shape[0],2))*(-1)

        for (tr_idx, cv_idx) in kfold.split(X,y):
            self._fit_one_fold(X.iloc[tr_idx])
            tmp = self._transform_one_fold(X.iloc[cv_idx])
            res[cv_idx] = tmp.values
        tmp = pd.DataFrame(res,\
            columns=['w_high_' + self.column_name,'w_med_' + self.column_name])
        X = pd.concat([X.reset_index(drop=True),tmp],axis=1)
        return X

    def fit_transform_test(self, Xtrain, Xtest):
        self._fit_one_fold(Xtrain)
        tmp = self._transform_one_fold(Xtest)
        Xtest = pd.concat([Xtest.reset_index(drop=True),tmp],axis=1)
        return Xtest

class EnsembleClassifiersTransformer():
    def __init__(self, classifiers=None, folds = 5):
        self.classifiers = classifiers
        self.kfold = StratifiedKFold(folds)

    def _fit_one_fold(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def _predict_one_fold(self, X):
        res = np.ones((X.shape[0],1))*(-1)
        for classifier in self.classifiers:
            res = np.column_stack((res,classifier.predict_proba(X)))
        return np.array(res[:,1:])

    def fit_transform_train(self, X, y):
        res = np.ones((X.shape[0],len(self.classifiers)*3))*(-1)
        # X_train = X.todense()
        X_train = X
        # k-fold for training set
        for (tr_idx, cv_idx) in self.kfold.split(X_train,y):
            X_tr,y_tr = X_train[tr_idx],y[tr_idx]
            X_cv,y_cv = X_train[cv_idx],y[cv_idx]
            self._fit_one_fold(X_tr,y_tr)
            res[cv_idx,:] = self._predict_one_fold(X_cv)
            print ("Fold results (cv error):")
            for (idx,clf) in enumerate(self.classifiers):
                print("clf {:2d}: {:06.4f}".\
                    format(idx, log_loss(y_cv,clf.predict_proba(X_cv))))
        return res

    def fit_transform_test(self, Xtr, ytr, Xts):
        # Xtr = Xtr.todense()
        # Xts = Xts.todense()
        self._fit_one_fold(Xtr,ytr)
        return self._predict_one_fold(Xts)

class MetaFeatures(BaseEstimator, TransformerMixin):
    def __init__(self,clf,folds=5,n_class=3):
        self.clf = clf
        self.kfold = StratifiedKFold(folds)
        self.n_class = n_class
        self.train_shape = None
        self.train_result = None

    def fit(self, X, y=None):
        self.train_shape = X.shape
        self.train_result = np.ones((X.shape[0],self.n_class))*(-1)
        X = X.todense()
        for (tr_idx, cv_idx) in self.kfold.split(X,y):
            X_tr,y_tr = X[tr_idx],y[tr_idx]
            X_cv,y_cv = X[cv_idx],y[cv_idx]
            self.clf.fit(X_tr,y_tr)
            self.train_result[cv_idx,:] = self.clf.predict_proba(X_cv)
        self.clf.fit(X)
        return self

    def transform(self, X):
        if X.shape == self.train_shape:
            res = self.train_result
        else:
            res = self.clf.predict_proba(X)
        res = sparse.hstack([X,sparse.csr_matrix(res)])
        return res

class MeanTargetTransformerNew():
    def __init__(self,group,folds=5,n_class=3):
        self.kfold = StratifiedKFold(folds)
        self.group = group
        self.name_low = "_".join(group)+'_low'
        self.name_med = "_".join(group)+'_med'
        self.name_hig = "_".join(group)+'_hig'
        self.names = [self.name_low,self.name_med,self.name_hig]

    def fit_transform_train(self, X, y):
        X[self.name_low] = np.nan
        X[self.name_med] = np.nan
        X[self.name_hig] = np.nan

        for (tr_idx, cv_idx) in self.kfold.split(X,y):
            print(cv_idx)
            X_tr = X.iloc[tr_idx]
            X_cv = X.iloc[cv_idx]
            tmp = X_tr.groupby(self.group+['interest_level']).size().\
                unstack(fill_value=np.nan).reset_index()
            tmp.rename(columns={\
                'low':self.name_low,\
                'medium':self.name_med,\
                'high':self.name_hig,},inplace=True)
            
            # normalize
            row_sum = np.nansum(\
                tmp[[self.name_low,self.name_med,self.name_hig]],axis=1)
            for name in self.names:
                tmp[name] = tmp[name]/row_sum
            
            # merge to fill on values on correct index for CV portion only
            res=pd.merge(X.iloc[cv_idx],tmp,\
                how='left',on=self.group,suffixes=('','_'))
            res.set_index(cv_idx,inplace=True)
            
            # fillna full set on index
            for name in self.names:
                X[name].fillna(res[name+'_'],inplace=True,axis='index')
            print(self.group,X[X[self.name_low].isnull()].shape[0])

        for name in self.names:
            X[name].fillna(0,inplace=True)

        return X

    def fit_transform_test(self, X, Xt):
        # group and rename
        tmp = X.groupby(self.group+['interest_level']).size().\
            unstack(fill_value=np.nan).reset_index()
        tmp.rename(columns={\
            'low':self.name_low,\
            'medium':self.name_med,\
            'high':self.name_hig,},inplace=True)

        # normalize
        row_sum =\
            np.nansum(tmp[[self.name_low,self.name_med,self.name_hig]],\
                axis=1)
        for name in self.names:
            tmp[name] = tmp[name]/row_sum

        Xt = pd.merge(Xt,tmp,on=self.group,how='left')
        return Xt

class MeanTargetTransformerOld():
    def __init__(self,group,folds=5,n_class=3):
        self.kfold = StratifiedKFold(folds)
        self.group = group
        self.name_low = group+'_low'
        self.name_med = group+'_med'
        self.name_hig = group+'_hig'
        self.names = [self.name_low,self.name_med,self.name_hig]

    def fit_transform_train(self, X, y):
        X[self.name_low] = np.nan
        X[self.name_med] = np.nan
        X[self.name_hig] = np.nan

        for (tr_idx, cv_idx) in self.kfold.split(X,y):
            X_tr = X.iloc[tr_idx]
            X_cv = X.iloc[cv_idx]

            for m in X_tr.groupby(self.group):
                idx = X_cv[X_cv[self.group] == m[0]].index
                X.loc[idx, self.name_low] =\
                    (m[1].interest_level == 'low').mean()
                X.loc[idx, self.name_med] =\
                    (m[1].interest_level == 'medium').mean()
                X.loc[idx, self.name_hig] =\
                    (m[1].interest_level == 'high').mean()

            self.res = X

        return X

    def fit_transform_test(self, X, Xt):
        Xt[self.name_low] = np.nan
        Xt[self.name_med] = np.nan
        Xt[self.name_hig] = np.nan
        for m in X.groupby(self.group):
            idx = Xt[Xt[self.group] == m[0]].index
            Xt.loc[idx, self.name_low] =\
                (m[1].interest_level == 'low').mean()
            Xt.loc[idx, self.name_med] =\
                (m[1].interest_level == 'medium').mean()
            Xt.loc[idx, self.name_hig] =\
                (m[1].interest_level == 'high').mean()
        self.res = Xt
        return Xt

class Debugger(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        print("Pipeline output shape: ",X.shape)
        return X

class MyMinMaxScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self.maxs = np.nanmax(X,axis=0)
        self.mins = np.nanmin(X,axis=0)
        self.scale = self.maxs-self.mins
        self.shape = X.shape[1]
        return self

    def transform(self, X):
        if X.shape[1] != self.shape:
            print("Error: shape mismatch")
        res = (X.values - self.mins)/self.scale[None,:]
        return res

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields
        
    def fit(self, X,y):
        return self
        
    def transform(self, X):
        return X[self.fields]

def get_importance(gbm):
    """
    Getting relative feature importance
    """
    importance = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
    importance = importance/1.e-2/importance.values.sum()
    return importance

def create_submission(score, pred, model, importance):
    """
    Saving model, features and submission
    """
    ouDir = '../output/'
    
    now = datetime.datetime.now()
    scrstr = "{:0.4f}_{}".format(score,now.strftime("%Y-%m-%d-%H%M"))
    
    mod_file = ouDir + '.model_' + scrstr + '.model'
    print('Writing model: ', mod_file)
    model.save_model(mod_file)
    
    sub_file = ouDir + 'submit_' + scrstr + '.csv'
    print('Writing submission: ', sub_file)
    pred.to_csv(sub_file, index=False)

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None,\
    seed_val=0, num_rounds=2000, max_depth=6, eta=0.01):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,\
            early_stopping_rounds=100)
    else:
        xgtest = xgb.DMatrix(test_X)
        watchlist = [ (xgtrain,'train') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
    
# misc function for cleaning features
def clean_features(row,cleaned_features):
    res = []
    for f in row:
        if f in cleaned_features:
            res.append("_".join(cleaned_features[f].split(" ")))
    return " ".join(res)

data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print("Starting shape of train: ",train_df.shape)
print("Starting shape of test: ",test_df.shape)

# Merge for feature processing
joint = pd.concat([train_df,test_df])

# --------------------------------
# conventional feature engineering
joint["room_dif"] = joint["bedrooms"]-joint["bathrooms"] 
joint["room_sum"] = joint["bedrooms"]+joint["bathrooms"] 
joint["bed_per_roomsum"] = joint["bedrooms"]/joint["room_sum"]
joint["num_photos"] = joint["photos"].apply(len)
joint["num_features"] = joint["features"].apply(len)
joint["num_description_words"] =\
    joint["description"].apply(lambda x: len(x.split(" ")))

# clusters and locations
ny_lat = 40.785091
ny_lon = -73.968285
joint['distance_from_center'] = haversine_np(ny_lon,ny_lat,\
    joint["longitude"],joint["latitude"])

# Normalize (longitude, latitude) before K-means
minmax = MinMaxScaler()
joint["lat_scaled"] = minmax.fit_transform(joint["latitude"].\
    values.reshape(-1,1))
joint["lon_scaled"] = minmax.fit_transform(joint["longitude"].\
    values.reshape(-1,1))

# Fit k-means and get labels
kmeans = KMeans(n_clusters=40)
kmeans.fit(joint.loc[joint["distance_from_center"]<12,\
    ['lon_scaled', 'lat_scaled']])
joint.loc[joint["distance_from_center"]<12,"kmeans40"] = kmeans.labels_

kmeans = KMeans(n_clusters=80)
kmeans.fit(joint.loc[joint["distance_from_center"]<12,\
    ['lon_scaled', 'lat_scaled']])
joint.loc[joint["distance_from_center"]<12,"kmeans80"] = kmeans.labels_

# prices
joint["price_per_bed"] = joint["price"]/joint["bedrooms"]
joint.loc[joint["bedrooms"] == 0, "price_per_bed"] = joint["price"]
joint["price_per_room"] = joint["price"]/\
    (1.0+0.5*joint["bathrooms"].clip(0,2)+joint["bedrooms"].clip(1,4))

# convert the created column to datetime object so as to extract more features 
joint["created"] = pd.to_datetime(joint["created"])
joint["created_date"] = joint["created"].dt.date.astype('datetime64')
joint["passed"] = joint["created"] - joint["created"].min()
joint["passed_days"] = joint.passed.dt.days
joint["passed_weeks"] = (joint["passed_days"]/7).astype('int')
joint["created_year"] = joint["created"].dt.year
joint["created_month"] = joint["created"].dt.month
joint["created_day"] = joint["created"].dt.day
joint["created_hour"] = joint["created"].dt.hour

# add description sentiment processed by R-syuzhet
sent_train = pd.read_json(data_path+'train_description_sentiment.json')
sent_train.rename(columns={'train_df$listing_id':'listing_id'},inplace=True)
sent_test = pd.read_json(data_path+'test_description_sentiment.json')
sent_test.rename(columns={'test_df$listing_id':'listing_id'},inplace=True)
sent_joint = pd.concat([sent_train,sent_test])
joint = pd.merge(joint,sent_joint,how='left',on='listing_id')

# --------------------------------
# Adding counts of listings by keys_to_count
keys_to_count = ["manager_id","building_id","display_address",
    "kmeans40","kmeans80"]
count_features = []

for key in keys_to_count:
    by_key = joint[["price",key]].groupby(key).count().reset_index()
    by_key.columns = [key,"listings_by_"+key]
    joint = pd.merge(joint,by_key,how='left',on=key)
    count_features.append("listings_by_"+key)

# --- Adding mean of keys by categorical features
keys_to_average = ["price","price_per_bed"]
grps_to_average = ["manager_id","building_id",\
    "display_address","street_address",\
    "kmeans40","kmeans80"]
mean_features = []

for key in keys_to_average:
    for grp in grps_to_average:
        by_grp = \
            joint[[key,grp]].groupby(grp).mean().reset_index()
        by_grp.rename(columns={key: key+'_by_'+grp},inplace=True)
        joint = pd.merge(joint,by_grp,how='left',on=grp)
        mean_features.append(key+'_by_'+grp)

mean_features.remove('price_by_street_address')

# --- Adding two level means
keys_to_average = ["price_per_room"]
grps_to_average = [
    ["manager_id","passed_days"],\
    ["building_id","passed_days"],\
    ["manager_id","building_id"]]

for key in keys_to_average:
    for grp in grps_to_average:
        name = key+'_by_'+grp[0]+"_"+grp[1]
        by_grp = \
            joint[[key,grp[0],grp[1]]].groupby(grp).mean().reset_index()
        by_grp.rename(columns={key: name},inplace=True)
        joint = pd.merge(joint,by_grp,how='left',on=grp)
        mean_features.append(name)

# --------------------------------
# Merge with exif
exif = pd.read_csv("../input/exif_digital.csv")

counter = []
for i in exif.columns:
    counter.append([i,exif[exif[i].notnull()].shape[0]])

counter = pd.DataFrame(counter,columns=["field","count"])
counter = counter[counter["count"]>10000]

exif_features = counter["field"].values.tolist()
exif_features.remove("listing_id")

joint = pd.merge(joint,exif[counter["field"].values],how="left",on="listing_id")

# --------------------------------
# Process features
joint['features'] =\
    joint["features"].apply(lambda x:\
    [i.lower().strip() for i in x])

# prepare deduplicated features
cleaned_features = defaultdict()
for f in pd.read_csv("feature_deduplication.csv").values.tolist():
    cleaned_features[f[0]] = f[1]

joint['features'] = joint['features'].apply(\
    lambda x: clean_features(x,cleaned_features))

# --------------------------------
# Process description
joint['description'] =\
    joint["description"].apply(lambda x:\
    " ".join([i.lower().strip() for i in x.split(" ") if len(i)>4]))

# --------------------------------
# Process categorical features
categorical = [\
    "display_address",\
    "manager_id",\
    "building_id",\
    "street_address",
    ]

# Clean addresses
joint["street_address"] = joint["street_address"].apply(lambda x:\
    x.lower().strip())
joint["display_address"] = joint["display_address"].apply(lambda x:\
    x.lower().strip())

# Remove entries with one record
for key in categorical:
    counts = joint[key].value_counts()
    joint.ix[joint[key].isin(counts[counts==1].index),key] = "-1"

# Apply LabelEncoder for Hot Encoding to work
joint[categorical] = joint[categorical].apply(LabelEncoder().fit_transform)

# Split back
train_df = joint[joint.interest_level.notnull()].copy()
test_df = joint[joint.interest_level.isnull()].copy()

# --------------------------------
# Price quantiles
bins = train_df['price_per_room'].quantile(np.arange(0.05, 1, 0.05))
train_df['price_per_room_quant'] = np.digitize(train_df['price_per_room'], bins)
test_df['price_per_room_quant'] = np.digitize(test_df['price_per_room'], bins)

bins = train_df['price'].quantile(np.arange(0.05, 1, 0.05))
train_df['price_quant'] = np.digitize(train_df['price'], bins)
test_df['price_quant'] = np.digitize(test_df['price'], bins)

# --------------------------------
# Categorical transformer
columns = ['manager_id','building_id']
cat_features = []
for col in columns:
    ctf = CategoricalTransformer(col)
    train_df = ctf.fit_transform_train(train_df,train_df["interest_level"])
    test_df = ctf.fit_transform_test(train_df,test_df)
    cat_features.append('w_high_'+col)
    cat_features.append('w_med_'+col)

# --------------------------------
# Mean Target transformer - one level
columns = [
    ['manager_id'],\
    ['building_id'],\
    ['manager_id','price_per_room_quant']]
target_mean_features = []
for col in columns:
    trf = MeanTargetTransformerNew(col)
    train_df = trf.fit_transform_train(train_df,train_df['interest_level'])
    test_df = trf.fit_transform_test(train_df,test_df)
    target_mean_features.append("_".join(col)+'_low')
    target_mean_features.append("_".join(col)+'_med')
    target_mean_features.append("_".join(col)+'_hig')
    # target_mean_features.append(col+'_low')
    # target_mean_features.append(col+'_med')
    # target_mean_features.append(col+'_hig')

# --------------------------------
# Basic numerical features for neural network
train_df,test_df = mlp_features(train_df,test_df,n_min=15,precision=3)

'''
===============================
Define features
===============================
'''

# define continuous features - will be untouched
simple_features = [\
    "listing_id",\
    "bathrooms", "bedrooms", "latitude", "longitude",\
    "price","price_per_bed","price_per_room",\
    "price_quant","price_per_room_quant",\
    "num_photos","num_features","num_description_words",\
    "created_month","created_day","created_hour",\
    "room_dif","room_sum"]

# sentiment features for additional pipeline
sentiment_features = [\
    "anger","anticipation","disgust","fear","joy","negative",\
    "positive","sadness","surprise","trust"]

# numerical from NN kernel for additional pipeline
nn_features = [i for i in train_df.columns.values if i.startswith('num_')]

'''
===============================
Define X & y
===============================
'''
target_num_map = {'high':0, 'medium':1, 'low':2}
y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
X = train_df
X_test = test_df

'''
===============================
Define Pipeline and Ensembles
===============================
'''

# Define pipelines
pipe1 = Pipeline([
    ('features', FeatureUnion([
        ('simple_scaler', Pipeline([
            ('get', ColumnExtractor(\
                simple_features+\
                count_features+\
                mean_features+\
                categorical+\
                nn_features)),
            ('debugger', Debugger())
        ])),
        ('target_mean', Pipeline([
            ('get', ColumnExtractor(target_mean_features)),
            ('debugger', Debugger())
        ])),
        ('apartment_features', Pipeline([
            ('get', ColumnExtractor("features")),
            ('transform', CountVectorizer(max_features=346)),
            ('debugger', Debugger())
        ]))
    ]))
])
params1 = {'num_rounds':3000,'max_depth':6,'eta':0.03}

'''
===============================
XGboost Cycle
===============================
'''
mode = 'Val'
pipeline = pipe1

if mode == 'Val':
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
    
    X_train = pipeline.fit_transform(X_train,y_train)
    X_val = pipeline.transform(X_val)

    preds, model = runXGB(X_train,y_train,X_val,y_val,\
        num_rounds=3000,max_depth=6,eta=0.03)

elif mode == 'Train':
    X_train = pipeline.fit_transform(X,y)
    X_test = pipeline.transform(X_test)

    preds, model = runXGB(X_train, y, X_test,\
        num_rounds=2000,max_depth=6,eta=0.03)

    # Prepare Submission
    out_df = pd.DataFrame(preds)
    out_df.columns = ["high", "medium", "low"]
    out_df["listing_id"] = test_df.listing_id.values
    create_submission(0.517063, out_df, model, None)