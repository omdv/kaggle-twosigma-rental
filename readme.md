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
0.535718 | 0.54225 | price by mngr, bldng, addrs, 867-1000 trees *** BEST ***
0.537745 |         | added norm price by mngr, bld, addr, 711
0.535518 | 0.54408 | added only norm price by mngr, 759 *** CV BEST ***
0.537324 |         | same as above but removed room_dif
0.536441 |         | presumably the same as my *** BEST *** but dropped CV?
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
[933] test-mlogloss:0.533139 no skill and w/onehot (X_shape = 28677) ***BEST***
#### With Manager Skill
[925] test-mlogloss:0.534963 with skill (to=5)
[925] test-mlogloss:0.534661 with skill (to=10)
[949] test-mlogloss:0.535035 with skill (to=13,75centile)
#### No Manager Skill
[1133] test-mlogloss:0.536042 with createdby onehotencoded
[923] test-mlogloss:0.534524 added passed days
[1251] test-mlogloss:0.534307 added 20 neighbourhoods from description
[1058] test-mlogloss:0.534832 added only 4 neighbourhoods

### Manager Skill Optimizations
- Changed mean for all values - no improvement
- Skill = 3*High + Medium - no improvement



### Cross-validation
CV | Iter | Size
0.550214 | 375 | 25%
0.555614 | 368 | 33%
0.547370 | 317 | 20%
0.548243 | 422 | 10%
0.553261 | 281 | 5%
0.574613 | 296 | 2%

### Ideas
Listings created by day, month, week
Price normalized by area
Apartment Features by area - how unique
Areas grouped by coordinates
Price by building, address, manager and ratio to current