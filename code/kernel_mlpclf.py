import numpy as np
import pandas as pd
import time as time

def mlp_features(df_train, df_test, n_min=50, precision=3):
    
    # Interest: Numerical encoding of interest level
    df_train['y'] = 0.0
    df_train.loc[df_train.interest_level=='medium', 'y'] = 1.0
    df_train.loc[df_train.interest_level=='high', 'y'] = 2.0
    
    # Location features: Latitude, longitude
    df_train['num_latitude'] = df_train.latitude.values
    df_test['num_latitude'] = df_test.latitude.values
    df_train['num_longitude'] = df_train.longitude.values
    df_test['num_longitude'] = df_test.longitude.values
    x = np.sqrt(((df_train.latitude - df_train.latitude.median())**2) + (df_train.longitude - df_train.longitude.median())**2)
    df_train['num_dist_from_center'] = x.values
    x = np.sqrt(((df_test.latitude - df_train.latitude.median())**2) + (df_test.longitude - df_train.longitude.median())**2)
    df_test['num_dist_from_center'] = x.values
    df_train['pos'] = df_train.longitude.round(precision).astype(str) + '_' + df_train.latitude.round(precision).astype(str)
    df_test['pos'] = df_test.longitude.round(precision).astype(str) + '_' + df_test.latitude.round(precision).astype(str)
    
    # Degree of "outlierness"
    OutlierAggregated = (df_train.bedrooms > 4).astype(float)
    OutlierAggregated2 = (df_test.bedrooms > 4).astype(float)
    OutlierAggregated += (df_train.bathrooms > 3).astype(float)
    OutlierAggregated2 += (df_test.bathrooms > 3).astype(float)
    OutlierAggregated += (df_train.bathrooms < 1).astype(float)
    OutlierAggregated2 += (df_test.bathrooms < 1).astype(float)
    x = np.abs((df_train.price - df_train.price.median())/df_train.price.std()) > 0.30
    OutlierAggregated += x.astype(float)
    x2 = np.abs((df_test.price - df_train.price.median())/df_train.price.std()) > 0.30
    OutlierAggregated2 += x2.astype(float)
    x = np.log1p(df_train.price/(df_train.bedrooms.clip(1,3) + df_train.bathrooms.clip(1,2))) > 8.2
    OutlierAggregated += x.astype(float)
    x2 = np.log1p(df_test.price/(df_test.bedrooms.clip(1,3) + df_test.bathrooms.clip(1,2))) > 8.2
    OutlierAggregated2 += x2.astype(float)
    x = np.sqrt(((df_train.latitude - df_train.latitude.median())**2) + (df_train.longitude - df_train.longitude.median())**2) > 0.30
    OutlierAggregated += x.astype(float)
    x2 = np.sqrt(((df_test.latitude - df_train.latitude.median())**2) + (df_test.longitude - df_train.longitude.median())**2) > 0.30
    OutlierAggregated2 += x2.astype(float)
    df_train['num_OutlierAggregated'] = OutlierAggregated.values
    df_test['num_OutlierAggregated'] = OutlierAggregated2.values
    
    # Average interest in unique locations at given precision
    x = df_train.groupby('pos')['y'].aggregate(['count', 'mean'])
    d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
    impute = df_train.y.mean()
    df_train['num_pos'] = df_train.pos.apply(lambda x: d.get(x, impute))
    df_test['num_pos'] = df_test.pos.apply(lambda x: d.get(x, impute))
    
    # Density in unique locations at given precision
    vals = df_train['pos'].value_counts()
    dvals = vals.to_dict()
    df_train['num_pos_density'] = df_train['pos'].apply(lambda x: dvals.get(x, vals.min()))
    df_test['num_pos_density'] = df_test['pos'].apply(lambda x: dvals.get(x, vals.min()))

    # Building null
    df_train['num_building_null'] = (df_train.building_id=='0').astype(float)
    df_test['num_building_null'] = (df_test.building_id=='0').astype(float)
    
    # Building supervised
    x = df_train.groupby('building_id')['y'].aggregate(['count', 'mean'])
    d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
    impute = df_train.y.mean()
    df_train['num_building_id'] = df_train.building_id.apply(lambda x: d.get(x, impute))
    df_test['num_building_id'] = df_test.building_id.apply(lambda x: d.get(x, impute))
    
    # Building frequency
    d = np.log1p(df_train.building_id.value_counts()).to_dict()
    impute = np.min(np.array(list(d.values())))
    df_train['num_fbuilding'] = df_train.building_id.apply(lambda x: d.get(x, impute))
    df_test['num_fbuilding'] = df_test.building_id.apply(lambda x: d.get(x, impute))
    
    # Manager supervised
    x = df_train.groupby('manager_id')['y'].aggregate(['count', 'mean'])
    d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
    impute = df_train.y.mean()
    df_train['num_manager'] = df_train.manager_id.apply(lambda x: d.get(x, impute))
    df_test['num_manager'] = df_test.manager_id.apply(lambda x: d.get(x, impute))

    # Manager frequency
    d = np.log1p(df_train.manager_id.value_counts()).to_dict()
    impute = np.min(np.array(list(d.values())))
    df_train['num_fmanager'] = df_train.manager_id.apply(lambda x: d.get(x, impute))
    df_test['num_fmanager'] = df_test.manager_id.apply(lambda x: d.get(x, impute))
    
    # Creation time features
    df_train['created'] = pd.to_datetime(df_train.created)
    df_train['num_created_weekday'] = df_train.created.dt.dayofweek.astype(float)
    df_train['num_created_weekofyear'] = df_train.created.dt.weekofyear
    df_test['created'] = pd.to_datetime(df_test.created)
    df_test['num_created_weekday'] = df_test.created.dt.dayofweek
    df_test['num_created_weekofyear'] = df_test.created.dt.weekofyear
    
    # Bedrooms/Bathrooms/Price
    df_train['num_bathrooms'] = df_train.bathrooms.clip_upper(4)
    df_test['num_bathrooms'] = df_test.bathrooms.clip_upper(4)
    df_train['num_bedrooms'] = df_train.bedrooms.clip_upper(5)
    df_test['num_bedrooms'] = df_test.bedrooms.clip_upper(5)
    df_train['num_price'] = df_train.price.clip_upper(10000)
    df_test['num_price'] = df_test.price.clip_upper(10000)
    bins = df_train.price.quantile(np.arange(0.05, 1, 0.05))
    df_train['num_price_q'] = np.digitize(df_train.price, bins)
    df_test['num_price_q'] = np.digitize(df_test.price, bins)
    
    # Composite features based on: 
    # https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths
    df_train['num_priceXroom'] = (df_train.price / (1 + df_train.bedrooms.clip(1, 4) + 0.5*df_train.bathrooms.clip(0, 2))).values
    df_test['num_priceXroom'] = (df_test.price / (1 + df_test.bedrooms.clip(1, 4) + 0.5*df_test.bathrooms.clip(0, 2))).values
    df_train['num_even_bathrooms'] = ((np.round(df_train.bathrooms) - df_train.bathrooms)==0).astype(float)
    df_test['num_even_bathrooms'] = ((np.round(df_test.bathrooms) - df_test.bathrooms)==0).astype(float)
    
    # Other features
    df_train['num_features'] = df_train.features.apply(lambda x: len(x))
    df_test['num_features'] = df_test.features.apply(lambda x: len(x))
    df_train['num_photos'] = df_train.photos.apply(lambda x: len(x))
    df_test['num_photos'] = df_test.photos.apply(lambda x: len(x))
    df_train['num_desc_length'] = df_train.description.str.split(' ').str.len()
    df_test['num_desc_length'] = df_test.description.str.split(' ').str.len()
    df_train['num_desc_length_null'] = (df_train.description.str.len()==0).astype(float)
    df_test['num_desc_length_null'] = (df_test.description.str.len()==0).astype(float)
    
    # Features/Description Features
    bows = {'nofee': ['no fee', 'no-fee', 'no  fee', 'nofee', 'no_fee'],
            'lowfee': ['reduced_fee', 'low_fee','reduced fee', 'low fee'],
            'furnished': ['furnished'],
            'parquet': ['parquet', 'hardwood'],
            'concierge': ['concierge', 'doorman', 'housekeep','in_super'],
            'prewar': ['prewar', 'pre_war', 'pre war', 'pre-war'],
            'laundry': ['laundry', 'lndry'],
            'health': ['health', 'gym', 'fitness', 'training'],
            'transport': ['train', 'subway', 'transport'],
            'parking': ['parking'],
            'utilities': ['utilities', 'heat water', 'water included']
          }
    for fname, bow in bows.items():
        x1 = df_train.description.str.lower().apply(lambda x: np.sum([1 for i in bow if i in x]))
        x2 = df_train.features.apply(lambda x: np.sum([1 for i in bow if i in ' '.join(x).lower()]))
        df_train['num_'+fname] = ((x1 + x2) > 0).astype(float).values
        x1 = df_test.description.str.lower().apply(lambda x: np.sum([1 for i in bow if i in x]))
        x2 = df_test.features.apply(lambda x: np.sum([1 for i in bow if i in ' '.join(x).lower()]))
        df_test['num_'+fname] = ((x1 + x2) > 0).astype(float).values

    return df_train, df_test