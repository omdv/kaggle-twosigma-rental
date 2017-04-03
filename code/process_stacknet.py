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
	"../output/submission_itislit.csv",
	"../output/stacknet_submission.csv",
	"../output/submit_0.5213_2017-03-26-1745.csv",
	"../output/submit_0.5222_2017-04-02-1101.csv",
	"../output/submit_0.5229_2017-03-20-2150.csv",
	"../output/submit_0.5242_2017-04-02-1238.csv",
	"../output/submit_0.5244_2017-03-18-2255.csv",
	"../output/submit_0.5246_2017-04-02-1115.csv",
	"../output/submit_0.5256_2017-04-02-1257.csv",
	"../output/submit_0.5273_2017-03-18-2248.csv",
	]
weights = np.ones(len(files))/len(files)

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

res["listing_id"]=res["listing_id"].astype("int")
res.to_csv("../output/submit_10_averaged.csv", index=False)