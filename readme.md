### Submission history
0.567000 | 0.57412 | no description, 200 features countVectorizer for features
0.566670 | 0.57347 | with description 200 features
0.551920 |         | with categorical, descripion, features and weekday
0.555259 | 0.56021 | with added sentiment
0.551055 | 0.55728 | without description, 200 features, 300 steps
0.565918 |         | with TFIDF description
0.550662 |         | without description, 300 features
0.548755 | 		   | 25% validation size, with normalized price
0.544738 | 0.55485 | 25% validation, norm price, joint features, 400 steps
0.544937 |         | 25% val, joint features, 250 vectorizer, 315 steps
0.545613 |         | 25% val, added distance from city center, 317 steps
0.548117 |         | reduced features, 200 sparse, 381 steps
0.544038 | 0.55456 | maxd=4, 0.8, 791, 200 sparse
0.545684 | 0.54778 | starter-03, testCV, 1000 iters, 758itersCV
0.546251 | 0.55186 | starter-03 with added image features
0.547777 |         | starter-03 with workday, 718 iters
0.554220 |         | starter-03 with description sentiment, 494 iters
0.548053 |         | starter-03 with image and weekday, 973 iters
0.554330 |         | starter-03 removed bottom 8, 800 iters
0.547408 |         | starter-03 with dist from center, 784iters
0.545684 | 0.54850 | starter-03, 758 iters based on best CV, worse than 1000
0.545523 |         | starter-03, eta 0.05, 1601 trees
0.543843 | 0.54998 | starter-03, eta 0.1, 1000 iters, with image height
0.543843 | 0.54933 | starter-03, eta 0.1, 1131 iters with image height
0.546097 |         | starter-03 with image brightness
0.544752 | 0.54757 | starter-03 with eta 0.02, 3652 iters
0.550213 |         | starter-03 without listing_id, 829 trees
0.545969 |         | refactor, 804 trees
0.538228 | 0.54336 | refactor with listings by manager and building, 666 trees
0.537686 |         | as above, added by address, 730 trees
0.539730 |         | added listings by day, 713 trees
0.535718 | 0.54225 | price by mngr, bldng, addrs, 867-1000 trees
0.537745 |         | added norm price by mngr, bld, addr, 711
0.535518 | 0.54408 | added only norm price by mngr, 759
0.537324 |         | same as above but removed room_dif
0.536441 |         | presumably the same as my best but dropped CV?
0.536627 |         | lat, long fixed
0.536340 |         | lat, long scaled
0.536312 |         | fixed lat/long, locs kde, gauss 5e-3, 830 iters
0.536765 | 0.54249 | scaled lat/long, locs kde, gauss 5e-3

### Pipelines version (0.33 CV)
[748] test-mlogloss:0.547256 - baseline xgboost-starter-03
[629] test-mlogloss:0.541889 - added number of listings
[680] test-mlogloss:0.53842 - added price by ...
[609] test-mlogloss:0.538005 - added skill (th=10)
[652] test-mlogloss:0.53716 - added skill (th=5)
[933] test-mlogloss:0.533139 no skill and w/onehot (X_shape = 28677)
[1114] test-mlogloss:0.533229 with mean bedrooms by categorical, X=20647
[911] test-mlogloss:0.536069 with mean bedrooms, X=27681
[1096] test-mlogloss:0.533035 new best CV after refactor
[1127] test-mlogloss:0.534379 lowercase addresses X=27676 --REMOVING--
[1196] test-mlogloss:0.535263 all single categories in one group --REMOVING--

#### With Manager Skill
[925] test-mlogloss:0.534963 with skill (to=5)
[925] test-mlogloss:0.534661 with skill (to=10)
[949] test-mlogloss:0.535035 with skill (to=13,75centile)
#### No Manager Skill
[1133] test-mlogloss:0.536042 with createdby onehotencoded
[923] test-mlogloss:0.534524 added passed days
[1251] test-mlogloss:0.534307 added 20 neighbourhoods from description
[1058] test-mlogloss:0.534832 added only 4 neighbourhoods
[969] test-mlogloss:0.536202 min-max-scaler for continuous
[1214] test-mlogloss:0.533714 with fixed prices
[923] test-mlogloss:0.533143 with log_price
#### Starting from 0.533139 CV
[1213] test-mlogloss:0.533972 with sorted_date --REMOVING--
[1015] test-mlogloss:0.53366 with fixed prices --REMOVING--
[1141] test-mlogloss:0.53483 with 20 neighbourhoods --REMOVING--
#### Continuous feature engineering
[735] test-mlogloss:0.554142 - BEST CV w only first pipeline
[764] test-mlogloss:0.553327 - added listings_by_street_address --KEEP--
[673] test-mlogloss:0.554342 - added price_by_street_address --REMOVE--
[695] test-mlogloss:0.553595 - added 20 districts --REMOVE--
[649] test-mlogloss:0.551930 - added mean bedrooms by categories --KEEP--, 27ft
[712] test-mlogloss:0.552559 - added mean room_sum by categories --REMOVE--
[734] test-mlogloss:0.553119 - mean price_per_bed by categories --REMOVE--
[734] test-mlogloss:0.553011 - mean num_photos by categories --REMOVE--
[646] test-mlogloss:0.553748 - mean price by created day --REMOVE--
[703] test-mlogloss:0.552404 - mean price per bed by created day --REMOVE--
[686] test-mlogloss:0.553217 - listings by created day --REMOVE--
[590] test-mlogloss:0.552183 - categorical with one record reduced to -1

#### Adding high-cardinality data encoding
[1999] test-mlogloss:0.547035 - switched off hot-encoding **BASELINE**
[1999] test-mlogloss:0.536914 - added building and manager encoding
[1999] test-mlogloss:0.527346 - same as above but grouped singletones
[1999] test-mlogloss:0.530126 - same as above but with hot encoding --REMOVE--
[1999] test-mlogloss:0.540653 - added display address --TRY WITH DIM REDUCTION--
[1999] test-mlogloss:0.54904 - lowering addresses --REMOVE FOR NOW--
[1999] test-mlogloss:0.528982 - hot encoding only addresses --REMOVE--
[741] test-mlogloss:0.524402 - manager, building, no singles, eta=0.1
[692] test-mlogloss:0.526064 - added label encoded categorical --REMOVE--
[855] test-mlogloss:0.524997 - added label encoded addresses --REMOVE--
[851] test-mlogloss:0.524472 - price_per_bath --REMOVE--
[682] test-mlogloss:0.515589 - weight encoding by bedrooms, overfit? --NO LB IMPROVEMENT--
[840] test-mlogloss:0.52529 - mean price_per_bed by categories --REMOVE--
[772] test-mlogloss:0.525989 - price_per_bed infinities - price *10 --REMOVE--
[720] test-mlogloss:0.524053 - price_per_bed = price when infinity --KEEP--
[868] test-mlogloss:0.524994 - added addresses, lowercase --REMOVE--
[759] test-mlogloss:0.526138 - added description sentiment --REMOVE--
[754] test-mlogloss:0.52314 - added mean price_per_bed --KEEP--
[842] test-mlogloss:0.522895 - new best CV after refactor, order of features --BEST LB--
[863] test-mlogloss:0.523752 - same as best but with deduplicated features
[777] test-mlogloss:0.525409 - added kmeans 40 clusters
[1442] test-mlogloss:0.521757 - averaging by kmeans40 and listings by kmeans --KEEP--
[2018] test-mlogloss:0.52133 - same but with kmeans80 --BEST CV--
[2122] test-mlogloss:0.522056 - with 256 clusters and mean encoding --REMOVE--
[1221] test-mlogloss:0.521398 - with 512 clusters mean encoded --REMOVE--
[1623] test-mlogloss:0.528774 - with 1024 mean encoded clusters --REMOVE--
[1821] test-mlogloss:0.530953 - with 350 mean encoded --REMOVE--
[456] train-mlogloss:0.396015	test-mlogloss:0.525469 - same as prev best CV but scaled and NA=-1 (pipe3)
[589] test-mlogloss:0.521896 - best single-model CV (scaled continuous with apartment features)
[460] test-mlogloss:0.521388 - new price per bed model
[555] test-mlogloss:0.520152 - with two-level means ('price_per_room_by_manager_id_passed_days',
 'price_per_room_by_building_id_passed_days')
[446] test-mlogloss:0.527186 - added kmeans80 into categorcal encoding --REMOVE--
[477] test-mlogloss:0.519664 - added manager+building to 2nd level means --BEST CV-- --BEST SINGLE-MODEL LB--
[508] test-mlogloss:0.521989 - same but if in dense format
[519] test-mlogloss:0.520803 - two level counts (three options with mng, bld and passed days), individual are worse --REMOVE--
[431] test-mlogloss:0.521486 - price per room divided by cat_encoded --REMOVE--
[569] test-mlogloss:0.527082 - with gdy5 features and my xgboost params
[121] test-mlogloss:0.517705 - with 700 xgb meta-feature
[139] test-mlogloss:0.515926 - with 700 xgb and rf as meta-features


#### Rejected features
- price_per_bath
- price_by_street_address

#### Image EXIF
'IPTC:Keywords' - 279 occurences
[1217] test-mlogloss:0.527858 - with all digital exif features (1012 features)
[1706] test-mlogloss:0.527062 - exif > 200 count (287 features)
[1633] test-mlogloss:0.526572 - exif > 1000 count (57 features)
[1298] test-mlogloss:0.526275 - exif > 3000 count (18)
[688] test-mlogloss:0.526999 - exif > 10000 count (11)

#### For averaging
[1731] test-mlogloss:0.522178 - with description sentiment
[842] test-mlogloss:0.522895 - new best CV after refactor, order of features --BEST LB--
[2018] test-mlogloss:0.52133 - same but with kmeans80 --BEST CV--
[1883] test-mlogloss:0.524587 - with exif
[1974] test-mlogloss:0.52419 - refactored factorizer with exif
[1696] test-mlogloss:0.525625 - with 400 description

#### MetaClassifier
[427]	train-mlogloss:0.421107	test-mlogloss:0.532785 - pipe1 validation
[790]	train-mlogloss:0.317533	test-mlogloss:0.502662 - pipe2 validation
##### Case 0
Pipe1: numerical only
clf1: [mlp,xgbc1,xgbc2,gbc,ada,lr,knbc]
[76] test-mlogloss:0.521289
clf  0: 0.5802
clf  1: 0.5343
clf  2: 0.5799
clf  3: 0.5444
clf  4: 1.0743
clf  5: 0.6068
clf  6: 0.6742
##### Case 1
Pipe1(numerical)
Pipe2(continuous)
clf: usual set
[85]	train-mlogloss:0.49138	test-mlogloss:0.513679
##### Case 2
Pipe1
Pipe2
Pipe3
Pipe4
clf: usual set
[76]	train-mlogloss:0.486024	test-mlogloss:0.508938
LB: 0.52809
##### Case 3
six pipelines
clf: usual set
[102]	train-mlogloss:0.469265	test-mlogloss:0.50607
LB: 0.52800


#### Mean Target features
[452]	train-mlogloss:0.41483	test-mlogloss:0.531843 - simple, manager_id, count, mean
[565]	train-mlogloss:0.386249	test-mlogloss:0.531189 - same but without scaler and assigning -1 to NAN
[1827]	train-mlogloss:0.398074	test-mlogloss:0.519171 - added apt features
[489]	train-mlogloss:0.393853	test-mlogloss:0.520365 - same with NAN=-1 and scaler
[866]	train-mlogloss:0.410159	test-mlogloss:0.521129 - my scaler
[1376]	train-mlogloss:0.392146	test-mlogloss:0.519128 - added label-encoded categorical
[921]	train-mlogloss:0.41478	test-mlogloss:0.519115 - added bayesian-encoded mngr and bldng
[457]	train-mlogloss:0.337372	test-mlogloss:0.518959 - same as above but with max_depth=6
[289]	train-mlogloss:0.261728	test-mlogloss:0.522675 - max_depth=8
[1121]	train-mlogloss:0.321955	test-mlogloss:0.517424 - max_depth=6, eta=0.05
[1960]	train-mlogloss:0.326363	test-mlogloss:0.517063 - max_depth=6, eta=0.03 - LB 0.529
[2999]	train-mlogloss:0.386571	test-mlogloss:0.519287 - max_depth=6, eta=0.01
[1966]	train-mlogloss:0.328511	test-mlogloss:0.517943 - added target by building (SEPARATE PIPE)
[1518]	train-mlogloss:0.327771	test-mlogloss:0.516821 - new BEST CV with NN params
[2999]	train-mlogloss:0.384459	test-mlogloss:0.518427 - with exif (SEPARATE PIPE)
[1820]	train-mlogloss:0.320246	test-mlogloss:0.516347 - added price and price_per_room quantiles
[1412]	train-mlogloss:0.332733	test-mlogloss:0.516039 - added building in mean transformer
[1425]	train-mlogloss:0.329512	test-mlogloss:0.516461 - added mean by price_quant (SEPARATE PIPE)
[1559]	train-mlogloss:0.324325	test-mlogloss:0.516042 - added mean by price_per_room_quant (SEPARATE PIPE)
[1422]	train-mlogloss:0.316658	test-mlogloss:0.516445 - new mean transformer with building and manager_id (identical to 0.516039 with OLD)
[1547]	train-mlogloss:0.300891	test-mlogloss:0.51553 - added mean by manager and price_per_room_quant (BEST SINGLE-MODEL LB 0.52702 best single model after zeroed test set)
[1731]	train-mlogloss:0.330203	test-mlogloss:0.516617 - same but not zeroing nans in train portion of mean_transformer (REMOVE)
[1441]	train-mlogloss:0.299266	test-mlogloss:0.516432 - added manager+price_quant (SEP PIPELINE)
[1342]	train-mlogloss:0.30632	test-mlogloss:0.515501 - added manager+bedrooms (SEP PIPELINE)
[1442]	train-mlogloss:0.29893	test-mlogloss:0.515141 - manager+kmeans80 (SEP PIPELINE)
[1253]	train-mlogloss:0.306091	test-mlogloss:0.51577 - added dist_mass_center and dist_city_center (REMOVE)
[1559]	train-mlogloss:0.30734	test-mlogloss:0.51506 - added manager_id+dist_mass_center_q (SEP PIPELINE)
[1693]	train-mlogloss:0.279047	test-mlogloss:0.513288 - cross mix of 11 parameters in mean transformer (best CV)
[1373]	train-mlogloss:0.309265	test-mlogloss:0.515135 - repeated building, manager and manager+price_per_room_quant (LB 0.52743)
[909]	train-mlogloss:0.290929	test-mlogloss:0.515479 - building, manager and manager+price_per_room_quant with dense format
[1463]	train-mlogloss:0.304401	test-mlogloss:0.516063 - same as above sparse with only 100 apartment features for stacknet
[1084]	train-mlogloss:0.323111	test-mlogloss:0.515197 - pipe2 (LB 0.52985)
[1586]	train-mlogloss:0.288519	test-mlogloss:0.51366 - pipe3 (LB 0.52856)
[1410]	train-mlogloss:0.263855	test-mlogloss:0.512393 - pipe0 (BEST CV)
[1179]	train-mlogloss:0.280151	test-mlogloss:0.512727 - added dist_city_center, mass_center and passed_days (BEST CV) (LB 0.53019)
[1288]	train-mlogloss:0.294688	test-mlogloss:0.503187 - pipe1 with img_data
[1584]	train-mlogloss:0.26772	test-mlogloss:0.502468 - pipe6 with img_data(day, hour, month) (BEST CV, LB 0.51275)
[1328]	train-mlogloss:0.283729	test-mlogloss:0.502321 - pipe7 with same (LB 0.51423)
[1448]	train-mlogloss:0.30309	test-mlogloss:0.505918 - pipe8
[603]	train-mlogloss:0.31071	test-mlogloss:0.530282 - pipe6 
[953]	train-mlogloss:0.285217	test-mlogloss:0.50971 - dense pipe6 with new numerical features
[860]	train-mlogloss:0.297881	test-mlogloss:0.509088 - sparse pipe6 with new numerical features (20% CV)
[908]	train-mlogloss:0.291526	test-mlogloss:0.508328 - sparse pipe6 with old features (20% CV)
[963]	train-mlogloss:0.281361	test-mlogloss:0.510269 - sparse pipe6 with only cap letters
[1030]	train-mlogloss:0.269957	test-mlogloss:0.508093 - sparse pipe6 with manager_id+building_id
[880]	train-mlogloss:0.292271	test-mlogloss:0.508411 - added price_quant
[923]	train-mlogloss:0.290436	test-mlogloss:0.510132 - 0.025 in quants
[782]	train-mlogloss:0.313256	test-mlogloss:0.511976 - 0.1 in quants
[1822]	train-mlogloss:0.289718	test-mlogloss:0.509001 - sparse pipe6 with 0.015 eta
[1632]	train-mlogloss:0.314329	test-mlogloss:0.508074 - max_depth=5,eta=0.025 with dist_city_center_q
[1439]	train-mlogloss:0.334425	test-mlogloss:0.509569 - same without dist_city_center_q
[1719]	train-mlogloss:0.304406	test-mlogloss:0.506357 - same with buildin_id+manager_id
[1603]	train-mlogloss:0.316485	test-mlogloss:0.506921 - dense above


#### Stacking - best single model
pipe1:
    ['manager_id'],\
    ['building_id'],\
    ['manager_id','price_per_room_quant']]
[1547]	train-mlogloss:0.300891	test-mlogloss:0.51553

pipe2:
    ['manager_id','dist_mass_center_q'],
    ['manager_id','price_quant'],
    ['manager_id','bedrooms'],
    ['manager_id','kmeans80']]
[1574]	train-mlogloss:0.318671	test-mlogloss:0.51968

Three pipes stacking with 1600 estimators
Fold results (cv error):
clf  0: 0.5389
Fold results (cv error):
clf  0: 0.5575
Fold results (cv error):
clf  0: 0.5545
Fold results (cv error):
clf  0: 0.5550
Fold results (cv error):
clf  0: 0.5584
[763]	train-mlogloss:0.472375	test-mlogloss:0.515451

#### Second level ensemble
Four clf on top of first level, no improvement in CV
[86]	train-mlogloss:0.49731	test-mlogloss:0.507028

#### Final Stacking
8 pipelines
xgboost
pipe 3: [840]	train-mlogloss:0.310354	test-mlogloss:0.533529
pipe 4: [922]	train-mlogloss:0.294674	test-mlogloss:0.525568
pipe 5: [864]	train-mlogloss:0.257132	test-mlogloss:0.506677
pipe 6: [878]	train-mlogloss:0.280826	test-mlogloss:0.503798
pipe 7: [815]	train-mlogloss:0.288507	test-mlogloss:0.503288
pipe 8: [872]	train-mlogloss:0.274393	test-mlogloss:0.501187

lgbm (max_depth = -1, leaves = 31)
pipe 8: [995]	valid_0's multi_logloss: 0.505942 (max_depth = -1, leaves = 31) (BEST)
pipe 8: [951]	valid_0's multi_logloss: 0.506222 (max_depth = 6, leaves = 31)
pipe 8: [1054]	valid_0's multi_logloss: 0.506667 (max_depth = 12, leaves = 31)
pipe 8: [620]	valid_0's multi_logloss: 0.506197 (max_depth = -1, leaves = 50)
pipe 8: [1637]	valid_0's multi_logloss: 0.505668 (max_depth = -1, leaves = 20)
pipe 7: [802]	valid_0's multi_logloss: 0.507104
pipe 6: [943]	valid_0's multi_logloss: 0.508158
pipe 5: [904]	valid_0's multi_logloss: 0.505904
pipe 4: [949]	valid_0's multi_logloss: 0.522013
pipe 3: [743]	valid_0's multi_logloss: 0.537344
pipe 2: [975]	valid_0's multi_logloss: 0.530801

rfc
pipe 6: [1000] 0.5464419615
pipe 6: [800] 0.5485363897
pipe 6: [1200] 0.547966225968

#### XGBOOST with meta-feature
pipe 6: [1345]	train-mlogloss:0.283956	test-mlogloss:0.507502
pipe 6: [1041]	train-mlogloss:0.307376	test-mlogloss:0.509017
pipe 7: [814]	train-mlogloss:0.283525	test-mlogloss:0.504385
pipe 8

#### Best params:
xgbc1 depth=4, n_estim=500
xgbc2 depth=10, n_estim=200
rfc1 n_estim=1000 (200,500,1000 - 1000 is the best)
lr n_iter=300 (300 to 500 - no change)
gbc n_estim=700 (500 to 700 - 700 is the best)
knbc n_neighbors=128 (32, 64, 128, 256, 512 with 128 is the best)
gpc - hangs up
ada n_estimators=200 - (200,500,700,1000 - 200 is the best)
mlp - best is 'tanh' with 10,60,5

### Cross-validation
CV | Iter | Size
0.550214 | 375 | 25%
0.555614 | 368 | 33%
0.547370 | 317 | 20%
0.548243 | 422 | 10%
0.553261 | 281 | 5%
0.574613 | 296 | 2%

### Ideas
Do something with addresses
Do something with date
Do something with description
Listings created by day, month, week
Price normalized by area
Apartment Features by area - how unique
Areas grouped by coordinates
Price by building, address, manager and ratio to current