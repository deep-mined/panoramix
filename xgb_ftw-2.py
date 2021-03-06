# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%
TS_SPLIT = 10 # how many splits for timeseries
WINDOW_SIZE = 5 # how many minutes to aggregate as single datapoint
PREDICT_AHEAD1 = 6 # how far time ahead predict 1st model
PREDICT_AHEAD2 = 10 # how far ahead predict 2nd model
EARLY_STOP = 10 # stopping xgboost training if it's not progressing
RESAMPLE_MIN = 1 # original data are in 1-sec intervals, but that's not good for training?

# %%
data_folder = Path('../data/zad2-dane')


# %%
types = ['manipulowane', 'zaklocajace', 'zaklocane', 'straty']
types2 = ['man', 'zak', 'zcne', 'stra']
files_by_type = {} 
for t,t2 in zip(types,types2):
    files_by_type[t2]= sorted(list(data_folder.glob('2021-05-*/' + t + '*')))  # TODO: 2021-04-19 seems to not work


# %%
def all_types_to_df(files_dict, end=100):
    type_to_df = {}
    for t in types2:
        types_csv = []
        for p in files_dict[t][:end]:
            data_df = pd.read_csv(p)
            data_df.columns = map(lambda x: x.lower(), data_df.columns)
            data_df.columns = [data_df.columns[0]] + [t+'_'+nom for nom in data_df.columns[1:]]
            types_csv.append(data_df)
        type_to_df[t] = types_csv
    return type_to_df


# %%
type_df = all_types_to_df(files_by_type)

# %%
for t in types2:
    d = type_df[t][0].columns.values
    print(type_df[t][0].columns)
    for i in type_df[t]:
        assert np.all(i.columns.values == d), (i,d)

# %%
type_per_df = { t:pd.concat(type_df[t], ignore_index=True) for t in types2}

# %%
from functools import reduce
df_final = reduce(lambda left,right: pd.merge(left,right,on='czas'), type_per_df.values())

# %%
del df_final['man_unnamed: 5']
del df_final['zcne_unnamed: 5']

# %%
translate_naming = {
    "man_001fcx00285_sppv.pv":'man_air_flow',
    "man_001xxxcalc01.num.pv[3]":"man_co2",
    "man_001scx00274_sppv.pv":"man_blow",
    "man_001fcx00241_sppv.pv":"man_dust",
    "zak_001fyx00206_spsum.pv":"zak_mixer",
    "zak_001fcx00231_sppv.pv":"zak_fry",
    "zak_001fcx00251_sppv.pv":"zak_slag",
    "zak_001fcx00281.pv":"zak_oxy1",
    "zak_001fcx00262.pv":"zak_oxy2",
    "zak_001fcx00261.pv":"zak_air",
    "zak_001xxxcalc01.num.pv[2]":"zak_sturoxy",
    "zak_prob_corg":"zak_carbon",
    "zak_prob_s":"zak_sulfur",
    "zak_sita_nadziarno":"zak_over_seed",
    "zak_sita_podziarno":"zak_under_seed",
    "zak_poziom_zuzel":"zak_slag_level",
    "zcne_001ucx00274.pv":"zcne_angle",
    "zcne_001nir0ods0.daca.pv":"zcne_loss",
    "zcne_temp_zuz":"zcne_temp",
    "zcne_007sxr00555.daca1.pv":"zcne_shake",
    "stra_001nir0szr0.daca.pv": "stra_sum",
    "stra_001nir0szrg.daca.pv": "stra_1",
    "stra_001nir0s600.daca.pv":"stra_2",
    "stra_001nir0s500.daca.pv":"stra_3",
    "stra_001nir0s300.daca.pv":"stra_4",
    "stra_001nir0s100.daca.pv":"stra_5",
}
new_cols = [translate_naming.get(cname) or cname for cname in df_final.columns]
df_final.columns = new_cols
df_final = df_final.drop(['stra_1','stra_2','stra_3','stra_4','stra_5'], axis=1, errors='ignore')

# %% [markdown]
# # Resampling

# %%
df_final['czas'] = pd.to_datetime(df_final['czas'])
df_final = df_final.resample(f'{RESAMPLE_MIN}min', on='czas').mean()
df_final['stra_sum'] = df_final['stra_sum'].rolling(RESAMPLE_MIN).mean().shift(-RESAMPLE_MIN)
df_final = df_final[:-RESAMPLE_MIN]
df_final

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import xgboost as xgb

# %% [markdown]
# # Predict using data window of X1 minutes for X2 minutes ahead

# %%
X = df_final.iloc[:,:-1]
X_org = X.copy()
y = df_final.iloc[:,-1]
y_org = y.copy()

for i in range(1, WINDOW_SIZE):
    X_shifted1 = X_org.shift(-i).rename(index=None, columns=lambda c: c + f'_{i}')
    X = X.merge(X_shifted1, left_index=True, right_index=True)
X = X.iloc[:-i,:]
y = y.iloc[i:]
    
X = X.iloc[:-PREDICT_AHEAD1,:]
y = y.shift(-PREDICT_AHEAD1).iloc[:-PREDICT_AHEAD1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# %%
tscv = TimeSeriesSplit(TS_SPLIT)
model3 = xgb.XGBRegressor()
m = None
train_rmse = []
test_rmse = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    m = model3.fit(X_train, y_train, eval_metric='rmse', eval_set=[(X_train, y_train), (X_test,  y_test)], verbose=False, xgb_model = m, early_stopping_rounds=EARLY_STOP)
    train_rmse += model3.evals_result()['validation_0']['rmse']
    test_rmse += model3.evals_result()['validation_1']['rmse']

plt.figure(figsize=(25,10))
plt.plot(train_rmse, label='train')
plt.plot(test_rmse, label='test')

# %%
predictions3 = model3.predict(X_test)
print(mean_squared_error(predictions3, y_test))

# %%
predictions3 = model3.predict(X_test)
d = pd.DataFrame(predictions3, index=y_test.index, columns=['predykcja'])
d['orygina??'] = y_test
d.plot(figsize=(25,15))

# %% [markdown]
# # Predict using data window of X1 minutes for X2' minutes ahead

# %%
X = df_final.iloc[:,:-1]
X_org = X.copy()
y = df_final.iloc[:,-1]
y_org = y.copy()

for i in range(1, WINDOW_SIZE):
    X_shifted1 = X_org.shift(-i).rename(index=None, columns=lambda c: c + f'_{i}')
    X = X.merge(X_shifted1, left_index=True, right_index=True)
X = X.iloc[:-i,:]
y = y.iloc[i:]
    
X = X.iloc[:-PREDICT_AHEAD2,:]
y = y.shift(-PREDICT_AHEAD2).iloc[:-PREDICT_AHEAD2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# %%
tscv = TimeSeriesSplit(TS_SPLIT)
model4 = xgb.XGBRegressor()
m = None
train_rmse = []
test_rmse = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    m = model4.fit(X_train, y_train, eval_metric='rmse', eval_set=[(X_train, y_train), (X_test,  y_test)], verbose=False, xgb_model = m, early_stopping_rounds=EARLY_STOP)
    train_rmse += model4.evals_result()['validation_0']['rmse']
    test_rmse += model4.evals_result()['validation_1']['rmse']

plt.figure(figsize=(25,10))
plt.plot(train_rmse, label='train')
plt.plot(test_rmse, label='test')

# %%
predictions4 = model4.predict(X_test)
print(mean_squared_error(predictions4, y_test))

# %%
predictions4 = model4.predict(X_test)
d = pd.DataFrame(predictions4, index=y_test.index, columns=['predykcja'])
d['orygina??'] = y_test
d.plot(figsize=(25,15))


# %% [markdown]
# # Using models for predictions

# %%
def cartesian_product(*arrays):
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [len(arrays)], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, len(arrays))


# %%
# Prediction with windowed models

SETTINGS_CHANGE_UNIT = [80.0, 0.8, 2.0, 0.5]  # how hard it is to adjust param
ADJUSTMENT_TARGET_WEIGHT = 0.0 # how important is adjustment cost against target difference

def find_winning_params2(state, current_settings, target):
    # params configuration which we want to try
    params = cartesian_product(
        np.linspace(1900, 3500, 20), 
        np.linspace(65, 81, 81-65),
        np.linspace(40, 70, 70-40),
        np.linspace(13, 27, 27-13)
    ).round()
    aab = np.ones((params.shape[0], state.shape[1])) * state.T
    aab[:, 0:4] = params

    predictions3 = model3.predict(aab) # predict 6 min ahead
    predictions4 = model4.predict(aab) # predict 10 min ahead
    predictions = (predictions3 + predictions4) / 2

    target_diff = np.abs(predictions - target)
    adjustment_cost = np.sum(np.abs(aab[:, 0:4] - current_settings.reshape((1, 4))) / SETTINGS_CHANGE_UNIT, axis=1)
    winner = np.argmin(target_diff + adjustment_cost * ADJUSTMENT_TARGET_WEIGHT)

    return params[winner,:]  # settings which we believe are the best to obtain target from current 


# %%
X = df_final.iloc[:,:-1]
X_org = X.copy()

for i in range(1, WINDOW_SIZE):
    X_shifted1 = X_org.shift(-i).rename(index=None, columns=lambda c: c + f'_{i}')
    X = X.merge(X_shifted1, left_index=True, right_index=True)
X = X.iloc[:-i,:]
X = X.iloc[:-PREDICT_AHEAD2, :]

state = X.iloc[0].values.reshape((-1, 1)) # real-time input from sensors
current_settings = X.iloc[0, 0:4].values
print(current_settings)
for t in range(15, 25):
    print(t, find_winning_params2(state, current_settings, t))
    
state = X.iloc[4242].values.reshape((-1, 1)) # real-time input from sensors
current_settings = X.iloc[4242, 0:4].values
print(current_settings)
for t in range(15, 25):
    print(t, find_winning_params2(state, current_settings, t))
    
state = X.iloc[8888].values.reshape((-1, 1)) # real-time input from sensors
current_settings = X.iloc[8888, 0:4].values
print(current_settings)
for t in range(15, 25):
    print(t, find_winning_params2(state, current_settings, t))
