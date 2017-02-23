import os
import sys
import operator
import numpy as np
import pandas as pd
import pickle
import geocoder
import xgboost as xgb
from scipy import sparse
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors.kde import KernelDensity

def get_importance(gbm):
    """
    Getting relative feature importance
    """
    importance = pd.Series(gbm.get_fscore()).sort_values(ascending=False)
    importance = importance/1.e-2/importance.values.sum()
    return importance

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None,\
    seed_val=0,num_rounds=2000):
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
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest,ntree_limit=model.best_iteration+1)
    return pred_test_y, model
    
# --- Read data
data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

features_to_use  = ["lat_fixed", "lon_fixed", "price"]
joint = pd.concat([train_df,test_df])

# # Fixing coordinates
# # remove between from addresses
# joint['display_address'] = joint.display_address.str.replace('between.*','')

# # now find the coordinates for missing records
# mask = ((joint.longitude == 0) |\
#     (joint.latitude == 0) |\
#     (joint.latitude < 25) |\
#     (joint.longitude < -100))
# missingCoords = joint[mask]
# missingGeoms = (missingCoords.street_address + ' New York').\
#     apply(geocoder.google)
# joint['lat_fixed'] = joint.latitude
# joint['lon_fixed'] = joint.longitude
# joint.loc[mask, 'lat_fixed'] = missingGeoms.apply(lambda x: x.lat)
# joint.loc[mask, 'lon_fixed'] = missingGeoms.apply(lambda x: x.lng)

# # fix occurences of bad street_address
# mask = (joint.lon_fixed.isnull())
# missingCoords = joint[mask]
# missingGeoms = (missingCoords.display_address + ' New York').\
#     apply(geocoder.google)
# joint.loc[mask, 'lat_fixed'] = missingGeoms.apply(lambda x: x.lat)
# joint.loc[mask, 'lon_fixed'] = missingGeoms.apply(lambda x: x.lng)

# # coordinates KDE
# coords = joint[['lat_fixed','lon_fixed']]
# kde = KernelDensity(kernel='gaussian', bandwidth=5e-3).fit(coords)
# joint['locations_kde'] = kde.score_samples(coords)

coords = pd.read_csv('coordinates.csv')
joint = pd.merge(joint,coords,how='left',on='listing_id')
joint['locations_kde'] = np.exp(joint.locations_kde)

# scaling coordinates
scaler = preprocessing.MinMaxScaler()
joint['lat_scaled'] = scaler.fit_transform(joint.lat_fixed.reshape(-1,1))
joint['lon_scaled'] = scaler.fit_transform(joint.lon_fixed.reshape(-1,1))

# transformation of lat and lng #
joint["price_t"] = joint["price"]/joint["bedrooms"] 

joint["room_dif"] = joint["bedrooms"]-joint["bathrooms"] 
joint["room_sum"] = joint["bedrooms"]+joint["bathrooms"] 
joint["price_t1"] = joint["price"]/joint["room_sum"]
joint["fold_t1"] = joint["bedrooms"]/joint["room_sum"]

# count of photos #
joint["num_photos"] = joint["photos"].apply(len)

# count of "features" #
joint["num_features"] = joint["features"].apply(len)

# count of words present in description column #
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

# 0.547777
# train_df["created_weekday"] = train_df["created"].dt.weekday
# test_df["created_weekday"] = test_df["created"].dt.weekday

# 0.547408
# ny_lat = 40.785091
# ny_lon = -73.968285
# train_df['distance_from_center'] = np.sqrt((train_df.latitude-ny_lat)**2+\
#     (train_df.longitude-ny_lon)**2)
# test_df['distance_from_center'] = np.sqrt((test_df.latitude-ny_lat)**2+\
#     (test_df.longitude-ny_lon)**2)

# --- Adding image props - does not work
train = pd.read_pickle('train.pickled')
test = pd.read_pickle('test.pickled')
joint_pickled = pd.concat([train,test])

to_choose = ['img_brg_mean','img_hgt_mean','img_sat_mean','img_wdt_mean',\
'listing_id','compound','neg','pos','neu']
joint = pd.merge(joint,joint_pickled[to_choose],how='left',on='listing_id')

train = 0
test = 0
joint_pickled = 0

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
    joint[['price','display_address']].groupby('display_address')\
        .count().reset_index()
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
    joint[['price','display_address']].groupby('display_address')\
        .mean().reset_index()
by_address.columns = ['display_address','price_by_address']
joint = pd.merge(joint,by_address,how='left',on='display_address')

# 0.54408 best with only manager
# joint["price_by_address_norm"] = joint["price_by_address"]/joint["price"]/1.0
# joint["price_by_building_norm"] = joint["price_by_building"]/joint["price"]/1.0
# joint["price_by_manager_norm"] = joint["price_by_manager"]/joint["price"]/1.0


# --- Managers skill
managers = train_df[['manager_id','interest_level']]
interest_dummies = pd.get_dummies(managers.interest_level)
managers = pd.concat([managers,interest_dummies[['low','medium','high']]],\
    axis = 1).drop('interest_level', axis = 1)
managers = managers.groupby('manager_id').mean().reset_index()
managers['manager_skill'] = 2*managers['high']+managers['medium']


# adding all these new features to use list #
features_to_use.extend(["price_t", "num_features",\
    "num_description_words", "created_day",\
    "created_hour",'price_t1',"listing_id",\
    "listings_by_building","listings_by_manager","listings_by_address",\
    "price_by_building","price_by_manager","price_by_address"])

features_to_use.extend(['img_brg_mean','img_hgt_mean','img_sat_mean',\
    'img_wdt_mean','compound','pos','neu'])

features_to_use.extend(['locations_kde'])


categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
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


tfidf = CountVectorizer(stop_words='english', max_features=400)
tr_sparse = tfidf.fit_transform(train_df["proc_features"])
te_sparse = tfidf.transform(test_df["proc_features"])

# Change to add feature processing
train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()
# train_X = train_df[features_to_use]
# test_X = test_df[features_to_use]

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level']\
    .apply(lambda x: target_num_map[x]))
print(train_X.shape, test_X.shape)

cv_scores = []
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(range(train_X.shape[0])):
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break
        
# preds, model = runXGB(train_X, train_y, test_X, num_rounds=1000)
# out_df = pd.DataFrame(preds)
# out_df.columns = ["high", "medium", "low"]
# out_df["listing_id"] = test_df.listing_id.values
# out_df.to_csv("../output/xgb_starter_04_mods.csv", index=False)      

print(features_to_use)