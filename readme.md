### Submission history
CV | LB | comments
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
0.545684 | 0.54850 | starter-03, 758 iters based on best CV
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

0.539575 |         | ['bathrooms', 'bedrooms', 'lat_fixed', 'lon_fixed', 'price', 'price_t', 'num_photos', 'num_features', 'num_description_words', 'created_day', 'created_hour', 'price_t1', 'listing_id', 'listings_by_building', 'listings_by_manager', 'listings_by_address', 'price_by_building', 'price_by_manager', 'price_by_address', 'img_brg_mean', 'img_hgt_mean', 'img_sat_mean', 'img_wdt_mean', 'compound', 'pos', 'neu', 'neg', 'locations_kde', 'display_address', 'manager_id', 'building_id', 'street_address']

0.541679 | 0.54353 | switched back to my code, same as best cv with old code, 0.374612 test with 999 trees, starter-03 gives 0.375619 of train error and 0.54316 on LB
0.353673, my code CV: 0.541679, LB: 0.54353

### Simple
0.545684 | 0.54778 | baseline


### Debugging a difference between starter-03 and my code
starter-03:
no-sparse, 999, train error 0.384395
sparse, 999, train error 0.375619

no-sparse features
listing_id               7.766921
manager_id               6.930740
latitude                 6.522207
price                    6.431421
longitude                5.729030
price_by_manager         5.492510
num_description_words    5.227322
street_address           5.146093
price_t1                 5.126980
listings_by_manager      5.074420
display_address          4.993191
building_id              4.501039
price_by_address         4.183291
price_by_building        4.130731
price_t                  3.583630
listings_by_address      3.122536
num_features             2.943355
created_day              2.866904
listings_by_building     2.811955
created_hour             2.811955
num_photos               2.391476
bathrooms                0.611606
bedrooms                 0.468261
room_dif                 0.434814
room_sum                 0.387032
created_month            0.310581

my:
no-sparse, 999, 0.384658 

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