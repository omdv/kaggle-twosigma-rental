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
[519] test-mlogloss:0.520803 - two level counts (three options with mng, bld and passed days), individual are worse --REMOVE--


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


#### Second level ensemble
Four clf on top of first level, no improvement in CV
[86]	train-mlogloss:0.49731	test-mlogloss:0.507028


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