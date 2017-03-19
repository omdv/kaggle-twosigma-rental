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
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

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

class Debugger(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        print X.shape
        return X

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, fields):
        self.fields = fields
        
    def fit(self, X,y):
        return self
        
    def transform(self, X):
        return X[self.fields]

class ApartmentFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_features = 400):
        self.num_features = num_features
        
    def fit(self, X,y):
        self.tfidf = CountVectorizer(stop_words='english',\
            max_features=self.num_features)
        
        self.tfidf.fit(X['features'])
        return self
        
    def transform(self, X):
        X_sparse = self.tfidf.transform(X['features'])
        return X_sparse

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

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None,\
    seed_val=0,
    num_rounds=2000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 4
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
    
data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

# Merge for feature processing
joint = pd.concat([train_df,test_df])

# --------------------------------
# conventional feature engineering
joint["room_dif"] = joint["bedrooms"]-joint["bathrooms"] 
joint["room_sum"] = joint["bedrooms"]+joint["bathrooms"] 
joint["price_per_bed"] = joint["price"]/joint["bedrooms"]
joint["price_per_bath"] = joint["price"]/joint["bathrooms"]
joint["price_per_room"] = joint["price"]/joint["room_sum"]
joint["bed_per_roomsum"] = joint["bedrooms"]/joint["room_sum"]
joint["num_photos"] = joint["photos"].apply(len)
joint["num_features"] = joint["features"].apply(len)
joint["num_description_words"] =\
    joint["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
joint["created"] = pd.to_datetime(joint["created"])
joint["passed"] = joint["created"] - joint["created"].min()
joint["passed_days"] = joint.passed.dt.days
joint["created_year"] = joint["created"].dt.year
joint["created_month"] = joint["created"].dt.month
joint["created_day"] = joint["created"].dt.day
joint["created_hour"] = joint["created"].dt.hour

# Transform addresses
# joint["street_address"] = joint["street_address"].apply(lambda x:\
#     x.lower().strip())
# joint["display_address"] = joint["display_address"].apply(lambda x:\
#     x.lower().strip())

# --------------------------------
# Adding counts
by_manager = \
    joint[['price','manager_id']].groupby('manager_id').count().reset_index()
by_manager.columns = ['manager_id','listings_by_manager']
joint = pd.merge(joint,by_manager,how='left',on='manager_id')

by_building = \
    joint[['price','building_id']].groupby('building_id').count().reset_index()
by_building.columns = ['building_id','listings_by_building']
joint = pd.merge(joint,by_building,how='left',on='building_id')   

by_display_address = \
    joint[['price','display_address']].groupby('display_address').count().\
    reset_index()
by_display_address.columns = ['display_address','listings_by_display_address']
joint = pd.merge(joint,by_display_address,how='left',on='display_address')

by_street_address = \
    joint[['price','street_address']].groupby('street_address').count().\
    reset_index()
by_street_address.columns = ['street_address','listings_by_street_address']
joint = pd.merge(joint,by_street_address,how='left',on='street_address')

# --- Adding mean of keys
keys = ['price']
mean_features = []
for key in keys:
    by_manager = \
        joint[[key,'manager_id']].groupby('manager_id').mean().reset_index()
    by_manager.columns = ['manager_id',key+'_by_manager']
    joint = pd.merge(joint,by_manager,how='left',on='manager_id')

    by_building = \
        joint[[key,'building_id']].groupby('building_id').mean().reset_index()
    by_building.columns = ['building_id',key+'_by_building']
    joint = pd.merge(joint,by_building,how='left',on='building_id')   

    by_display_address = \
        joint[[key,'display_address']].groupby('display_address').mean().\
        reset_index()
    by_display_address.columns = ['display_address',key+'_by_display_address']
    joint = pd.merge(joint,by_display_address,how='left',on='display_address')

    by_street_address = \
        joint[[key,'street_address']].groupby('street_address').mean().\
        reset_index()
    by_street_address.columns = ['street_address',key+'_by_street_address']
    joint = pd.merge(joint,by_street_address,how='left',on='street_address')

    mean_features += [key+'_by_manager',key+'_by_building',\
        key+'_by_display_address',key+'_by_street_address']
mean_features.remove('price_by_street_address')

# adding price by created day
by_created_day = \
    joint[['listing_id','passed_days']].groupby('passed_days').count().\
    reset_index()
by_created_day.columns = ['passed_days','listings_by_created_day']
joint = pd.merge(joint,by_created_day,how='left',on='passed_days')

# Feature processing for CountVectorizer
joint['features'] =\
    joint["features"].apply(lambda x:\
    " ".join(["_".join(i.split(" ")) for i in x]))

# --------------------------------
# Process categorical features
categorical = [\
    "display_address",\
    "manager_id",\
    "building_id",\
    "street_address",
    ]

# Remove entries with one record
for key in categorical:
    counts = joint[key].value_counts()
    joint.ix[joint[key].isin(counts[counts==1].index),key] = "-1"

# Apply LabelEncoder for Hot Encoding to work
joint[categorical] = joint[categorical].apply(LabelEncoder().fit_transform)

# Split back
train_df = joint[joint.interest_level.notnull()]
test_df = joint[joint.interest_level.isnull()]

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

'''
===============================
Define features
===============================
'''

# define continuous features - will be untouched
continuous = [\
    "listing_id",\
    "bathrooms", "bedrooms", "latitude", "longitude", "price",\
    "price_per_bed","price_per_room",\
    "num_photos", "num_features","num_description_words",\
    "created_month","created_day","created_hour",\
    "room_dif","room_sum",\
    "listings_by_building","listings_by_manager",\
    "listings_by_display_address",\
    ]
continuous += mean_features
continuous += cat_features
# continuous += categorical

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
Define Pipeline
===============================
'''

# Define pipeline
NO_CHANGE_FIELDS = continuous
CATEGORY_FIELDS = ["street_address","display_address"]
TEXT_FIELDS = ["features"]
AVERAGING_FIELDS = ["manager_id","building_id"]

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('continuous', Pipeline([
            ('get', ColumnExtractor(NO_CHANGE_FIELDS)),
            ('debugger', Debugger())
        ])),
        # ('categorical', Pipeline([
        #     ('get', ColumnExtractor(CATEGORY_FIELDS)),
        #     ('onehot', OneHotEncoder(handle_unknown='ignore')),
        #     ('debugger', Debugger())
        # ])),
        ('vectorizer', Pipeline([
            ('get', ColumnExtractor(TEXT_FIELDS)),
            ('transform', ApartmentFeaturesVectorizer()),
            ('debugger', Debugger())
        ]))
    ]))
])


'''
===============================
XGboost Cycle
===============================
'''
Validation = False

if Validation:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
    
    X_train = pipeline.fit_transform(X_train,y_train)
    X_val = pipeline.transform(X_val)

    preds, model = runXGB(X_train,y_train,X_val,y_val)

else:
    X_train = pipeline.fit_transform(X,y)
    X_test = pipeline.transform(X_test)

    preds, model = runXGB(X_train, y, X_test, num_rounds=1000)

    # Prepare Submission
    out_df = pd.DataFrame(preds)
    out_df.columns = ["high", "medium", "low"]
    out_df["listing_id"] = test_df.listing_id.values
    create_submission(0.524402, out_df, model, None)