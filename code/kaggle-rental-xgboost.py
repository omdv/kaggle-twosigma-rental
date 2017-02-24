from __future__ import print_function
import pandas as pd
import numpy as np
import time
import copy
import datetime
import xgboost as xgb
import pickle
import exifread
from PIL import Image
from scipy import sparse
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import DBSCAN
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from subprocess import check_output

np.random.seed(2016)

def read_test_train(ifValid,valsize = 0.25):
    inputDir = "../input/"

    print("Load train.json...")
    train = pd.read_json(inputDir + "train.json")

    print("Load test.json...")
    test = pd.read_json(inputDir + "test.json")

    # create target class
    print("Preprocess dataset...")
    target_num_map = {'high':0, 'medium':1, 'low':2}
    train['target'] =\
        train['interest_level'].apply(lambda x: target_num_map[x])

    # split in validation set
    if ifValid:
        mask = np.random.rand(train.shape[0]) < valsize/1.e2
        valid = train[mask]
        train = train[~mask]
        valid['validation'] = 1.0
    else:
        valid = 0

    return train, test, valid

# --------------------------------------------------------#

def intersect(a, b):
    """
    For get_features() use
    """
    return list(set(a) & set(b))

# add derived features here
def derive_features(ifPickled,ifValid,train,test,valid):
    """
    This function allows making new features
    """
    print("Deriving new features...")

    # make a new joint for processing
    if ifValid:
        joint = pd.concat([train,test,valid])
    else:
        joint = pd.concat([train,test])

    # pickle contains image processing and description sentiment
    if ifPickled:
        joint_pickled = pd.read_pickle('joint.pickled')

    # --- Get coordinates
    coords = pd.read_csv('coordinates.csv')
    joint = pd.merge(joint,coords,how='left',on='listing_id')
    joint['locations_kde'] = np.exp(joint.locations_kde)

    # kms_per_radian = 6371.0088
    # epsilon = 0.5 / kms_per_radian
    # db = DBSCAN(eps=epsilon,\
    #     min_samples=1,\
    #     algorithm='ball_tree',\
    #     metric='haversine').fit(np.radians(coords[['lat_fixed','lon_fixed']]))
    # labels = db.labels_
    # n_clusters = len(set(labels))
    # clusters = pd.Series([coords[labels == n] for n in range(n_clusters)])
    # print('Number of clusters: {}'.format(n_clusters))

    # --- Scaling coordinates
    scaler = MinMaxScaler()
    joint['lat_scaled'] = scaler.fit_transform(joint.lat_fixed.reshape(-1,1))
    joint['lon_scaled'] = scaler.fit_transform(joint.lon_fixed.reshape(-1,1))

    # --- Get conventional features
    joint["price_t"] = joint["price"]/joint["bedrooms"] 
    joint["room_dif"] = joint["bedrooms"]-joint["bathrooms"] 
    joint["room_sum"] = joint["bedrooms"]+joint["bathrooms"] 
    joint["price_t1"] = joint["price"]/joint["room_sum"]
    joint["fold_t1"] = joint["bedrooms"]/joint["room_sum"]
    joint["num_photos"] = joint["photos"].apply(len)
    joint["num_features"] = joint["features"].apply(len)
    joint["num_description_words"] =\
        joint["description"].apply(lambda x: len(x.split(" ")))

    # --- Get date features
    joint["created"] = pd.to_datetime(joint["created"])
    joint["passed"] = joint["created"].max() - joint["created"]
    joint["created_year"] = joint["created"].dt.year
    joint["created_month"] = joint["created"].dt.month
    joint["created_day"] = joint["created"].dt.day
    joint["created_hour"] = joint["created"].dt.hour

    # --- Process features for sparse
    joint['proc_features'] = joint["features"].apply(lambda x:\
        " ".join(["_".join(i.split(" ")) for i in x]))

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

    joint["price_by_address_norm"] = joint["price_by_address"]/joint["price"]/1.0
    joint["price_by_building_norm"] = joint["price_by_building"]/joint["price"]/1.0
    joint["price_by_manager_norm"] = joint["price_by_manager"]/joint["price"]/1.0

    # --- Categorical
    categorical = [\
        "display_address",\
        "manager_id",\
        "building_id",\
        "street_address"]
    for f in categorical:
        if joint[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(joint[f].values))
            joint[f] = lbl.transform(list(joint[f].values))

    # # --- Managers skill
    # managers = train_df[['manager_id','interest_level']]
    # interest_dummies = pd.get_dummies(managers.interest_level)
    # managers = pd.concat([managers,interest_dummies[['low','medium','high']]],\
    #     axis = 1).drop('interest_level', axis = 1)
    # managers = managers.groupby('manager_id').mean().reset_index()
    # managers['manager_skill'] = 2*managers['high']+managers['medium']

    # --- Description Processing ---
    # tokenize description and get sentiment
    # print("Process description...")
    # joint['description_tokens'] =\
    #     joint['description'].apply(sent_tokenize)
    # joint = pd.concat([joint,\
    #     joint['description_tokens'].apply(description_sentiment)],axis=1)

    # --- Photo Processing ---
    # extract photo filenames from photos
    # print("Process photos...")
    # joint['photo_files'] =\
    #     joint['photos'].apply(lambda row: [x[29:] for x in row])
    # joint = joint.apply(process_all_images,axis=1)

    # split back
    test = joint[joint.interest_level.isnull()]
    if ifValid:
        train = joint[joint.interest_level.notnull() & joint.validation.isnull()]
        valid = joint[joint.validation.notnull()]
    else:
        train = joint[joint.interest_level.notnull()]
        valid = 0

    return train,test,valid


def get_features(train, test):
    """
    Get intersection of train and test as a list of features
    """
    features = intersect(train, test)
    # remove overfitting features
    toremove = ["photos",\
        "created",\
        "description",\
        "description_tokens",\
        "features",\
        "features_split",\
        "interest_level",\
        "target"]
    features = [x for x in features if x not in toremove]

    features_to_use=["bathrooms","bedrooms","lat_fixed","lon_fixed", "price"]
    features_to_use.extend([\
    "num_photos", "num_features","num_description_words",\
    "created_month", "created_day","created_hour",\
    "room_dif","room_sum","price_t1","price_t","listing_id",\
    "listings_by_building","listings_by_manager","listings_by_address",\
    "price_by_building","price_by_manager","price_by_address"])
    features_to_use.extend([\
        "display_address","manager_id","building_id","street_address"])

    features = features_to_use
    return sorted(features)

def add_sparse_features(train,test,valid,features,ifValid):
    print("Get sparse features...")

    # parse apartment features
    tfidf_ap = CountVectorizer(stop_words='english', max_features=400)
    if ifValid:
        tfidf_ap = tfidf_ap.fit(\
            list(train["proc_features"])+list(valid["proc_features"]))
    else:
        tfidf_ap = tfidf_ap.fit(list(train["proc_features"]))

    te_ap_sp = tfidf_ap.transform(test["proc_features"])
    tr_ap_sp = tfidf_ap.transform(train["proc_features"])
    if ifValid:
        va_ap_sp = tfidf_ap.transform(valid["proc_features"])
    dtrain = sparse.hstack([train[features], tr_ap_sp]).tocsr()
    dtest = sparse.hstack([test[features], te_ap_sp]).tocsr()
    if ifValid:
        dvalid = sparse.hstack([valid[features], va_ap_sp]).tocsr()
    else:
        dvalid = 0

    return dtrain, dtest, dvalid

# --------------------------------------------------------#

def run_single(dtrain,dtest,dvalid,ytrain,yvalid,\
    num_rounds,ifValid,seed_val=0):
    """
    Single xgboost run, returns predictions and validation score
    """

    # create a small validation set - unique people id
    print('Starting xgboost cycle...')
    print('Number of features: {}'.format(dtrain.shape[1]))

    print('Creating DMatrices...')
    dtest = xgb.DMatrix(dtest,missing = np.nan)
    if ifValid:
        dtrain = xgb.DMatrix(dtrain,label = ytrain,missing=np.nan)
        dvalid = xgb.DMatrix(dvalid,label = yvalid,missing=np.nan)
    else:
        dtrain = xgb.DMatrix(dtrain,label = ytrain)

    # tree booster params
    early_stopping_rounds = 50
    start_time = time.time()
    params = {}
    params['objective'] = 'multi:softprob'
    params['eta'] = 0.1
    params['max_depth'] = 4
    params['silent'] = 1
    params['num_class'] = 3
    params['eval_metric'] = "mlogloss"
    params['min_child_weight'] = 1
    params['subsample'] = 0.8
    params['colsample_bytree'] = 0.8
    params['seed'] = seed_val

    print('XGBoost params: {}'.format(params))

    if ifValid:
        watchlist = [(dtrain, 'train'),(dvalid,'val')]
    else:
        watchlist = [(dtrain, 'train')]

    gbm = xgb.train(params,dtrain,
        num_rounds, 
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=True,
        maximize=False)

    print('Validating...')
    if ifValid:
        check = gbm.predict(dvalid, ntree_limit=gbm.best_iteration+1)
        # score = log_loss(y_valid,check,labels=[0,1,2])
    else:
        check = gbm.predict(dtrain, ntree_limit=gbm.best_iteration+1)
        # score = log_loss(y_train,check,labels=[0,1,2])

    score = gbm.best_score
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm)
    #print('Importance array:\n{}'.format(imp))

    print("Predict test dataset...")
    test_prediction = gbm.predict(dtest, ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time()\
        - start_time)/60, 2)))
    return test_prediction, score, gbm, imp

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
    
    imp_file = ouDir + 'imp_' + scrstr + '.csv'
    print('Writing features: ', imp_file)
    importance.to_csv(imp_file)
    
    sub_file = ouDir + 'submit_' + scrstr + '.csv'
    print('Writing submission: ', sub_file)
    pred.to_csv(sub_file, index=False)

if __name__ == '__main__':
    # setup
    ifSparse = True
    ifPickled = False
    ifValid = True
    valsize = 25
    target = ['target']

    train, test, valid = read_test_train(ifValid,valsize)
    train, test, valid = derive_features(ifPickled,ifValid,train,test,valid)

    features = get_features(train, test)

    print('Shape of train: {}'.format(train[features].shape))
    print('Shape of test: {}'.format(test[features].shape))
    if ifValid:
        print('Shape of valid: {}'.format(valid[features].shape))
    print('Regular features: {}'.format(len(features)))

    ytrain = train[target]
    yvalid = valid[target] if ifValid else 0

    if ifSparse:
        dtrain, dtest, dvalid =\
            add_sparse_features(train,test,valid,features,ifValid)
    else:
        dtrain = train[features]
        dtest = test[features]
        dvalid = [valid[features] if ifValid else 0]

    prediction, score, model, importance =\
       run_single(dtrain,dtest,dvalid,ytrain,yvalid,1000,ifValid)
    
    #prepare submission
    out_df = pd.DataFrame(prediction)
    out_df.columns = ["high", "medium", "low"]
    out_df["listing_id"] = test.listing_id.values
    #create_submission(score, out_df, model, importance)
