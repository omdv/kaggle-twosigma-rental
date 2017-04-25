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
	"../output/submit_0.5090_2017-04-24-2317.csv",#pipe6 with meta
	"../output/submit_average_04242017.csv", #best
	"../output/submit_meta_average_04252017.csv" #unknown meta
	# "../output/submit_0.5000_2017-04-24-2314.csv",
	# "../output/submit_0.5044_2017-04-24-2239.csv",
	# "../output/submit_0.5082_2017-04-24-1629.csv",
	# "../output/submit_0.5086_2017-04-23-2226.csv"
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
res.to_csv("../output/submit_final_average_04252017.csv", index=False)