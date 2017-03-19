class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, k=5.0, f=1.0, r_k=0.01, folds=5):
        self.k = k
        self.f = f
        self.r_k = r_k
        self.column_name = column_name
        self.folds = folds

    def _reset_fold(self):
        if hasattr(self, '_one_fold_mapping'):
            self._one_fold_mapping = {}
            self.glob_med = 0
            self.glob_high = 0

    def _reset_fit(self):
        if hasattr(self,'_fit_mapping'):
            self._fit_mapping = {}

    def _fit_one_fold(self, X, y):
        self._reset_fold()

        tmp = X.groupby([self.column_name, 'interest_level']).size().\
            unstack().reset_index()
        tmp = tmp.fillna(0)

        tmp['record_count'] = tmp['high'] + tmp['medium'] + tmp['low']
        tmp['high_share'] = tmp['high']/tmp['record_count']
        tmp['med_share'] = tmp['medium']/tmp['record_count']

        self.glob_high = tmp['high'].sum()/tmp['record_count'].sum()
        self.glob_med = tmp['medium'].sum()/tmp['record_count'].sum()

        # Get weight function
        tmp['lambda'] = 1.0/(1.0+np.exp(np.float32(tmp['record_count']-self.k).\
            clip(-self.k,self.k)/self.f))
        
        # Blending
        tmp['w_high_'+self.column_name] =\
            (1.0-tmp['lambda'])*tmp['high_share']+tmp['lambda']*self.glob_high
        tmp['w_med_'+self.column_name] =\
            (1.0-tmp['lambda'])*tmp['med_share']+tmp['lambda']*self.glob_med

        # Adding random noise
        tmp['w_high_' + self.column_name] = tmp['w_high_' + self.column_name]*\
            (1+self.r_k*(np.random.uniform(size = len(tmp))-0.5))
        tmp['w_med_' + self.column_name] = tmp['w_med_' + self.column_name]*\
            (1+self.r_k*(np.random.uniform(size = len(tmp))-0.5))

        self._one_fold_mapping = tmp[['w_high_' + self.column_name,\
            'w_med_' + self.column_name,  self.column_name]]
        return self

    def _transform_one_fold(self, X):
        X = pd.merge(X,self._one_fold_mapping,how='left',on=self.column_name)
        return X[['w_high_' + self.column_name,'w_med_' + self.column_name]]

    def fit(self, X, y):
        self._reset_fit()
        kfold = StratifiedKFold(self.folds)
        res = np.ones((X.shape[0],2))*(-1)

        for (tr_idx, cv_idx) in kfold.split(X,y):
            self._fit_one_fold(X.iloc[tr_idx],y[tr_idx])
            tmp = self._transform_one_fold(X.iloc[cv_idx])
            res[cv_idx] = tmp.values
        dd = pd.DataFrame(res,\
            columns=['w_high_' + self.column_name,'w_med_' + self.column_name])
        X = pd.concat([X,dd],axis=1)
        self._fit_mapping = X[['w_high_' + self.column_name,\
            'w_med_' + self.column_name,  self.column_name]]
        return self

    def transform(self, X):
        X = pd.merge(X,self._fit_mapping,how='left',on=self.column_name)
        # X['w_high_' + self.column_name] = X['w_high_' + self.column_name].\
        # apply(lambda x: x if not np.isnan(x) else self.glob_high)
        # X['w_med_' + self.column_name] = X['w_med_' + self.column_name].\
        # apply(lambda x: x if not np.isnan(x) else self.glob_med)
        return X[['w_high_' + self.column_name,'w_med_' + self.column_name]]