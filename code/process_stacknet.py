import numpy as np
import pandas as pd

# Merging stacknet results with listing id for submission
pred = np.loadtxt("sigma_stack_pred.csv",delimiter=",")
test = np.loadtxt("test_stacknet.csv",delimiter=",")
res = np.column_stack((pred,test[:,0]))
np.savetxt("../output/stacknet_submission.csv",\
	res,delimiter=",",fmt="%9.8f,%9.8f,%9.8f,%d",\
	header="high,medium,low,listing_id",comments='')

# Averaging with other scripts
sub1 = pd.read_csv("../output/submission_itislit.csv")
