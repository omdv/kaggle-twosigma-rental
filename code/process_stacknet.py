import numpy as np
import pandas as pd

# # Merging stacknet results with listing id for submission
# pred = np.loadtxt("../stacknet/sigma_stack_pred.csv",delimiter=",")
# test = np.loadtxt("../stacknet/test_stacknet.csv",delimiter=",")
# res = np.column_stack((pred,test[:,0]))
# np.savetxt("../output/stacknet_submission.csv",\
# 	res,delimiter=",",fmt="%9.8f,%9.8f,%9.8f,%d",\
# 	header="high,medium,low,listing_id",comments='')

# Averaging with other scripts
files = [\
	"../output/submission_itislit.csv",\
	"../output/stacknet_submission.csv",\
	"../output/submit_0.5229_2017-03-20-2150.csv"]
weights = [0.2,0.4,0.4]

data = []

for idx,file in enumerate(files):
	data.append(np.genfromtxt(file, dtype=float, delimiter=',', names=True))

for d in data:
	d.sort(order="listing_id")

# define the result array
res = np.zeros(data[0].shape[0],\
	dtype=[('high','f8'),('medium','f8'),('low','f8'),('listing_id','i4')])

for idx,weight in enumerate(weights):
	res["high"] += data[idx]["high"]*weight
	res["medium"] += data[idx]["medium"]*weight

res["low"] = 1.0 - res["high"]-res["medium"]
res["listing_id"] = data[0]["listing_id"]

res = pd.DataFrame(res)

# sub1.sort_values(by='listing_id',inplace=True)
# sub2.sort_values(by='listing_id',inplace=True)
# sub3.sort_values(by='listing_id',inplace=True)

# # res = sub2.copy()
# high = 0.2*sub1["high"]+0.4*sub2["high"]+0.4*sub3["high"]
# medium = 0.2*sub1["medium"]+0.4*sub2["medium"]+0.4*sub3["medium"]
# low = 1.0-high-medium
# listings = sub1["listing_id"].values

res["listing_id"]=res["listing_id"].astype("int")
res.to_csv("../output/stacknet_submission_averaged.csv", index=False)