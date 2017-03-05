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
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pipeline_manager_skill import ManagerSkill
from pipeline_apartment_features import ApartmentFeaturesVectorizer
from sklearn.pipeline import Pipeline

np.random.seed(42)

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
    num_rounds=1000):
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
            early_stopping_rounds=50)
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

joint = pd.concat([train_df,test_df])

# transformation of lat and lng #
joint["price_t"] = joint["price"]/joint["bedrooms"] 

joint["room_dif"] = joint["bedrooms"]-joint["bathrooms"] 
joint["room_sum"] = joint["bedrooms"]+joint["bathrooms"] 
joint["price_t1"] = joint["price"]/joint["room_sum"]
joint["fold_t1"] = joint["bedrooms"]/joint["room_sum"]

# counts
joint["num_photos"] = joint["photos"].apply(len)
joint["num_features"] = joint["features"].apply(len)
joint["num_description_words"] =\
    joint["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
joint["created"] = pd.to_datetime(joint["created"])
joint["passed"] = joint["created"].max() - joint["created"]

# Let us extract some features like year, month, day, hour from date columns #
joint["created_year"] = joint["created"].dt.year
joint["created_month"] = joint["created"].dt.month
joint["created_day"] = joint["created"].dt.day
joint["created_hour"] = joint["created"].dt.hour

# --- Adding counts
by_manager = \
    joint[['price','manager_id']].groupby('manager_id').count().reset_index()
by_manager.columns = ['manager_id','listings_by_manager']
joint = pd.merge(joint,by_manager,how='left',on='manager_id')

by_building = \
    joint[['price','building_id']].groupby('building_id').count().reset_index()
by_building.columns = ['building_id','listings_by_building']
joint = pd.merge(joint,by_building,how='left',on='building_id')   

by_address = \
    joint[['price','display_address']].groupby('display_address').count().\
    reset_index()
by_address.columns = ['display_address','listings_by_address']
joint = pd.merge(joint,by_address,how='left',on='display_address')

# --- Adding mean price
by_manager = \
    joint[['price','manager_id']].groupby('manager_id').mean().reset_index()
by_manager.columns = ['manager_id','price_by_manager']
joint = pd.merge(joint,by_manager,how='left',on='manager_id')

by_building = \
    joint[['price','building_id']].groupby('building_id').mean().reset_index()
by_building.columns = ['building_id','price_by_building']
joint = pd.merge(joint,by_building,how='left',on='building_id')   

by_address = \
    joint[['price','display_address']].groupby('display_address').mean().\
    reset_index()
by_address.columns = ['display_address','price_by_address']
joint = pd.merge(joint,by_address,how='left',on='display_address')

joint["price_by_address_norm"] = joint["price_by_address"]/joint["price"]/1.0
joint["price_by_building_norm"] = joint["price_by_building"]/joint["price"]/1.0
joint["price_by_manager_norm"] = joint["price_by_manager"]/joint["price"]/1.0


# define features
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

# adding all these new features to use list #
features_to_use.extend(["price_t","num_photos", "num_features",\
    "num_description_words","created_month", "created_day",\
    "created_hour",'room_dif','room_sum','price_t1',"listing_id",\
    "listings_by_building","listings_by_manager","listings_by_address",\
    "price_by_building","price_by_manager","price_by_address"])

# add proc_features for sparse pipeline, will be deleted in pipeline
features_to_use.extend(['proc_features'])


categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(joint[f].values))
            joint[f] = lbl.transform(list(joint[f].values))
            features_to_use.append(f)

# --- Feature processing
joint['proc_features'] =\
    joint["features"].apply(lambda x:\
        " ".join(["_".join(i.split(" ")) for i in x]))

train_df = joint[joint.interest_level.notnull()]
test_df = joint[joint.interest_level.isnull()]


# --- Define X and y
target_num_map = {'high':0, 'medium':1, 'low':2}
y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
X = train_df[features_to_use]
X_test = test_df[features_to_use]


'''
===============================
Cross-validation cycle
===============================
'''
# X_train, X_val, y_train, y_val =\
#     model_selection.train_test_split(X, y, test_size=0.33)

# # Define pipeline
# manager_skill = ManagerSkill()
# apt_feat = ApartmentFeaturesVectorizer()
# pipe = Pipeline([('apartment_features',apt_feat)])

# X_train = pipe.fit_transform(X_train,y_train)
# X_val = pipe.transform(X_val)

# # Run one cycle
# preds, model = runXGB(X_train,y_train,X_val,y_val)

'''
===============================
Main Fit cycle
===============================
'''

# Define pipeline
manager_skill = ManagerSkill()
apt_feat = ApartmentFeaturesVectorizer()
pipe = Pipeline([('manager_skill',manager_skill),\
    ('apartment_features',apt_feat)])

X_train = pipe.fit_transform(X,y)
X_test = pipe.transform(X_test)

preds, model = runXGB(X_train, y, X_test, num_rounds=1000)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
create_submission(model.best_score, out_df, model, None)
