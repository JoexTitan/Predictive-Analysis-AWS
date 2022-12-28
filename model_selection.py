import seaborn as sns
import pandas as pd
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.style.use('classic')

train_features_df.head()

train_target_df.head()

test_features_df.head()

train_merge = pd.merge(train_features_df, train_target_df, on='jobId')

train_merge.head()

train_merge.info()

train_merge.isnull().sum()

train_merge.describe(include = 'all')

train_merge = train_merge[train_merge.salary != 0]

train_merge.describe(include = 'all')

train_merge_corr_indx = train_merge.corr().index
plt.figure(figsize=(6,4))
g=sns.heatmap(train_merge[train_merge_corr_indx].corr(),annot=True,cmap="Blues")

def cat_boxplot(df, target, feature):
    df_feature_mean = df.groupby([feature],as_index=False).mean()
    df_feature_sorted = df_feature_mean.sort_values(by=target)
    plt.figure()
    sns.boxplot(x=feature, y=target, data=df, order=df_feature_sorted[feature].values)
    plt.xticks(rotation=45)

cat_features = ['companyId', 'jobType', 'degree', 'major', 'industry']
target = 'salary'
for feature in cat_features:
    cat_boxplot(train_merge, target, feature)

cat_features.remove('companyId')
num_features = ['yrsExp','distanceFromCity']
train_merge_Hot_Enc = train_merge[num_features].join(pd.get_dummies(train_merge[cat_features]))
train_merge_Hot_Enc.head()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(train_merge_Hot_Enc, train_merge['salary'], test_size=0.25, random_state=42)

from sklearn.model_selection import cross_val_score
def cv(model, features, target,fold): 
    Rcross = cross_val_score(model,features,target, cv=fold, scoring='neg_mean_squared_error')
    print(model)
    print('Mean: '+str(- Rcross.mean())+', Std: '+str(Rcross.std()))

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(train_x,train_y)

cv(lr,train_x,train_y,2)

test_x_predict = lr.predict(test_x)

ax1 = sns.distplot(test_y, hist=False, color="g", label="Actual Value")
sns.distplot(test_x_predict, hist=False, color="b", label="Estimated Value" , ax=ax1)

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=160, max_depth=5)

gbr.fit(train_x,train_y)

cv(gbr,train_x,train_y,2)

test_x_gbrpredict = gbr.predict(test_x)

ax1 = sns.distplot(test_y, hist=False, color="r", label="Real Values")
sns.distplot(test_x_gbrpredict, hist=False, color="b", label="Predicted Values" , ax=ax1)

fi = pd.Series(gbr.feature_importances_, index=train_merge_Hot_Enc.columns).nlargest(5).plot(kind='bar')
plt.xlabel('Feature importance')

test_features_df.drop(['companyId', 'jobId'],axis = 1)
test_features_df_Hot_Enc = test_features_df[num_features].join(pd.get_dummies(test_features_df[cat_features]))
test_features_df_Hot_Enc.head()

test_features_df_Hot_Enc_gbrpredict = gbr.predict(test_features_df_Hot_Enc)
test_features_df_Hot_Enc_gbrpredict

test_features_df_Hot_Enc_gbrpredict_df = pd.DataFrame(data=test_features_df_Hot_Enc_gbrpredict,  columns=["Predicted_salary"])

test_features_target_df = test_features_df.join(test_features_df_Hot_Enc_gbrpredict_df)
test_features_target_df