import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_df(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['route'] = df['pickup'] + '->' + df['delivery']
    df['distance_km'] = df['distance'] / 1000.0
    df['weight_k'] = df['weight'] / 1000.0
    df['is_long_haul'] = (df['distance'] > 1500).astype(int)
    return df

train = build_df('train-test.csv')
cutoff = train['date'].quantile(0.9)
tr = train[train['date'] <= cutoff].copy()
va = train[train['date'] > cutoff].copy()

features = ['pickup','delivery','distance','equipment','weight','market_index','quote_signal','month','dayofyear','weekofyear','is_long_haul','distance_km','weight_k','route']

numeric = ['distance','weight','market_index','quote_signal','month','dayofyear','weekofyear','is_long_haul','distance_km','weight_k']
cat = ['pickup','delivery','equipment','route']

def make_model(model):
    pre = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat)
    ])
    return Pipeline([('pre', pre), ('model', model)])

models = {
    'gbr': make_model(GradientBoostingRegressor(random_state=42)),
    'rf': make_model(RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    'hgb': make_model(HistGradientBoostingRegressor(random_state=42))
}

for name, model in models.items():
    model.fit(tr[features], tr['posted_rate'])
    pred = model.predict(va[features])
    mae = mean_absolute_error(va['posted_rate'], pred)
    rmse = mean_squared_error(va['posted_rate'], pred) ** 0.5
    r2 = r2_score(va['posted_rate'], pred)
    print(name, 'MAE:', round(mae, 2), 'RMSE:', round(rmse, 2), 'R2:', round(r2, 4))
