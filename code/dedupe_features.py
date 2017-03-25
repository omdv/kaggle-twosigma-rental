import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict, Counter

class FeatureProcessing():

	def __init__(self,features,cutoff=5):
		self.cutoff = cutoff
		self.features = features

	def clean(self,s):
	    x = s.replace("-", "")
	    x = x.replace(" ", "")
	    x = x.replace("twenty four hour", "24")
	    x = x.replace("24/7", "24")
	    x = x.replace("24hr", "24")
	    x = x.replace("24-hour", "24")
	    x = x.replace("24hour", "24")
	    x = x.replace("24 hour", "24")
	    x = x.replace("common", "cm")
	    x = x.replace("concierge", "doorman")
	    x = x.replace("bicycle", "bike")
	    x = x.replace("private", "pv")
	    x = x.replace("deco", "dc")
	    x = x.replace("decorative", "dc")
	    x = x.replace("onsite", "os")
	    x = x.replace("outdoor", "od")
	    x = x.replace("ss appliances", "stainless")
	    return x

	def feature_hash(self,x):
	    cleaned = clean(x, uniq)
	    key = cleaned[:4].strip()
	    return key

	def clean_features(self):
		feature_counts = Counter()
		for feature in self.features.features:
			feature_counts.update(feature)
		feature = sorted([k for (k,v) in feature_counts.items() if v > n])
		feature[:10]

train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")
train_test = pd.concat([train_df, test_df], 0)

features = train_test[["features"]].apply(
    lambda _: [list(map(str.strip, map(str.lower, x))) for x in _])

# count features and drop features with less than n counts

n = 5

feature_counts = Counter()
for feature in features.features:
    feature_counts.update(feature)
feature = sorted([k for (k,v) in feature_counts.items() if v > n])
feature[:10]

def clean(s):
    x = s.replace("-", "")
    x = x.replace(" ", "")
    x = x.replace("twenty four hour", "24")
    x = x.replace("24/7", "24")
    x = x.replace("24hr", "24")
    x = x.replace("24-hour", "24")
    x = x.replace("24hour", "24")
    x = x.replace("24 hour", "24")
    x = x.replace("common", "cm")
    x = x.replace("concierge", "doorman")
    x = x.replace("bicycle", "bike")
    x = x.replace("private", "pv")
    x = x.replace("deco", "dc")
    x = x.replace("decorative", "dc")
    x = x.replace("onsite", "os")
    x = x.replace("outdoor", "od")
    x = x.replace("ss appliances", "stainless")
    return x

def feature_hash(x):
    cleaned = clean(x, uniq)
    key = cleaned[:4].strip()
    return key

key2original = defaultdict(list)
k = 4
for f in feature:
    cleaned = clean(f)
    key = cleaned[:k].strip()
    key2original[key].append(f)

def to_tuples():
    for f in feature:
        key = clean(f)[:k].strip()
        yield (f, key2original[key][0])
        
deduped = list(to_tuples())
df = pd.DataFrame(deduped, columns=["original_feature", "unique_feature"])