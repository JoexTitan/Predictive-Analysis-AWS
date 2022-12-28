from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import numpy as np

class Data:
    def __init__(self, train_file, target_file, test_file, target, index):
        self.train = None
        self.test = None
        self.features = None
        self.target = target
        self.index = index

        self._load_data(train_file, target_file, test_file)
        self._clean_data()
        print(self.train.shape)
        print(self.test.shape)

    def encode_data(self):
        for col in self.features:
            group_dict = dict(self.train.groupby([col])[self.target].mean())
            self.train[col] = self.train[col].map(group_dict)
            self.test[col] = self.test[col].map(group_dict)

    def split_data(self):
        return self.train.drop(columns=self.target), self.train[self.target]
    
    def get_baseline(self):
        baseline_true = self.train[self.target].values.astype(float)
        mean_dict = dict(self.train.groupby(['industry'])[self.target].mean())
        baseline_pred = self.train.industry.map(mean_dict)
        baseline_mse = mean_squared_error(baseline_true, baseline_pred)
        print("Baseline: MSE=%.3f\n" % baseline_mse)
        return baseline_mse

    def _load_data(self, train_file, target_file, test_file):
        self.train = pd.read_csv(train_file)
        self.features = self.train.drop(columns=self.index).columns.values
        self.train = pd.merge(self.train, pd.read_csv(target_file), on=self.index)
        self.test = pd.read_csv(test_file)

    def _clean_data(self):
        self._drop_duplicates(self.train)
        self._drop_duplicates(self.test)
        self._drop_null(self.train)
        self._drop_null(self.test)
        self._check_col_validity(self.train, 'yrsExp', 0)
        self._check_col_validity(self.test,  'yrsExp', 0)
        self._check_col_validity(self.train, 'distanceFromCity', 0)
        self._check_col_validity(self.test,  'distanceFromCity', 0)
        self._check_col_validity(self.train, 'slr', 1)

    def _drop_duplicates(self, df):
        print(df.duplicated().sum())
        df.drop_duplicates(inplace=True)

    def _drop_null(self, df):
        invalid_jobs = df.index[df.isnull().sum(axis=1).gt(0)].values
        print(len(invalid_jobs))
        df.drop(index=invalid_jobs, inplace=True)

    def _fill_null(self, df):
        invalid_jobs = df.index[df.isnull().sum(axis=1).gt(0)].values
        print(len(invalid_jobs))
        df.fillna(df.mean(), inplace=True)
    
    def _check_col_validity(self, df, col, threshold):
        invalid_jobs = df.index[df[col].lt(threshold)]
        print((len(invalid_jobs), col))
        df.drop(index=invalid_jobs, inplace=True)

class FeatureEngineer(Data):
    def __init__(self, train_file, target_file, test_file, target, index):
        Data.__init__(self, train_file, target_file, test_file, target, index)
        self._stats = []

    def add_stats(self, cols, col_name):
        self._generate_stats(cols, col_name)
        self.train = self._merge_stats(self.train, cols)
        self.test = self._merge_stats(self.test,  cols)

    def _generate_stats(self, cols, col_name):
        group = self.train.groupby(cols)[self.target]
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        upper_bound = Q3 + 1.5 * (Q3 - Q1)
        
        self._stats = pd.DataFrame({col_name+"_mean" : group.mean()})
        self._stats[col_name + "_min"] = group.min()
        self._stats[col_name + "_Q1"] = Q1
        self._stats[col_name + "_median"] = group.median()
        self._stats[col_name + "_Q3"] = Q3
        self._stats[col_name + "_upper"] = upper_bound
        self._stats[col_name + "_max"] = group.max()
        
    def _merge_stats(self, df, cols):
        df = pd.merge(df, self._stats, on=cols, how='left')
        df.set_index(self.index, inplace=True)
        self._fill_null(df)
        return df
        

class Models:
    def __init__(self, data):
        self.train_X = data[0]
        self.train_y = data[1]
        self._models = []
        self._scores = []
        self._mse = []
        self._best_model = None
        self._best_mse = None
    
    def set_baseline(self, baseline):
        self._mse.append(baseline)

    def add_model(self, model):
        self._models.append(model)

    def cv_models(self, cv=None):
        for model in self._models:
            scores = cross_val_score(model, self.train_X, self.train_y, cv=cv,
                                     scoring='neg_mean_squared_error')
            mse = -1.0 * np.mean(scores)
            self._print_summary(model, scores, mse)

            if(mse <= min(self._mse)):
                self._best_model = model
                self._best_mse = mse
                
            self._scores.append(scores)
            self._mse.append(mse)

    def print_all(self):
        print("Models:", self._models)
        print("Scores:", self._scores)
        print("MSE:", self._mse)
        print()
        
    def print_best_model(self):
        print("Best model: %s\nBest MSE: %.3f" %
              (self._best_model, self._best_mse))
        self._best_model.fit(self.train_X, self.train_y)
        
        if hasattr(self._best_model, 'feature_importances_'):
            print("Feature Importances:")
            feature_importance = self._best_model.feature_importances_
            index = self.train_X.columns
            for i in range(len(feature_importance)):
                print("%s  %.6f" % (index[i], feature_importance[i]))
        print()
                
    def predict(self, test):
        test_pred = self._best_model.predict(test).round(3)
        return pd.DataFrame(test_pred, index=test.index, columns=[self.target])        

    def _print_summary(self, model, scores, mse):
        print("Model:", model)
        print("Scores:", *scores)
        print("MSE: %.3f" % mse)
        print()


if __name__ == "__main__":
    dataset = FeatureEngineer("../f_training_data.csv",
                              "../f_testing_data.csv",
                              "../target_training_data.csv")
    
    dataset.encode_data()

    features = ['companyId', 'jobType', 'degree', 'major', 'industry']
    dataset.add_stats(features, "CJDMI")
    
    model = Models(dataset.split_data())
    model.set_baseline(dataset.get_baseline())
    
    model.add_model(LinearRegression())
    model.add_model(Lasso())
    model.add_model(GradientBoostingRegressor(max_depth=8))
    model.add_model(Pipeline([("Scaler", StandardScaler()), 
                              ("GBR", GradientBoostingRegressor(max_depth=8))]))
    model.cv_models()
    model.print_all()
    model.print_best_model()
    filename = "test_salaries_prediction.csv"
    model.predict(dataset.test).to_csv(filename)
