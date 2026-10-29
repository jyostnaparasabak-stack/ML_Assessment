import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train = pd.read_csv('train-test.csv')
train['date'] = pd.to_datetime(train['date'])
train['month'] = train['date'].dt.month
train['day_of_year'] = train['date'].dt.dayofyear
train['week_of_year'] = train['date'].dt.isocalendar().week.astype(int)
train['route'] = train['pickup'] + '->' + train['delivery']
train['is_long_haul'] = (train['distance'] > 1500).astype(int)

features = ['pickup','delivery','distance','equipment','weight','month','day_of_year','week_of_year','route','is_long_haul']
X = train[features]
y = train['posted_rate']

cutoff = train['date'].quantile(0.8)
tr = train[train['date'] <= cutoff].copy()
va = train[train['date'] > cutoff].copy()

num = ['distance','weight','month','day_of_year','week_of_year','is_long_haul']
cat = ['pickup','delivery','equipment','route']

pre = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat)
])
model = Pipeline([('pre', pre), ('model', GradientBoostingRegressor(random_state=42))])
model.fit(tr[features], tr['posted_rate'])
pred = model.predict(va[features])
print('MAE', mean_absolute_error(va['posted_rate'], pred))
print('RMSE', mean_squared_error(va['posted_rate'], pred) ** 0.5)
print('R2', r2_score(va['posted_rate'], pred))
