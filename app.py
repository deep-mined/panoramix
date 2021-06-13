import streamlit as st
import pickle
import xgboost
import pandas as pd
from functools import reduce
from pathlib import Path
import glob
import numpy as np
import SessionState
## CONSTANTS

TS_SPLIT = 10 # how many splits for timeseries
WINDOW_SIZE = 5 # how many minutes to aggregate as single datapoint
PREDICT_AHEAD1 = 6 # how far time ahead predict 1st model
PREDICT_AHEAD2 = 10 # how far ahead predict 2nd model
EARLY_STOP = 10 # stopping xgboost training if it's not progressing
RESAMPLE_MIN = 1 # original data are in 1-sec intervals, but that's not good for training?

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

ss = SessionState.get(i=0)


st.title("Stabilizacja pieca zawiesinowego")


@st.cache
def load_sensor_data(data_folder: str) -> pd.DataFrame:
    data_folder = Path(data_folder)
    types = ['manipulowane', 'zaklocajace', 'zaklocane', 'straty']
    types2 = ['man', 'zak', 'zcne', 'stra']
    files_by_type = {} 
    for t,t2 in zip(types,types2):
        files_by_type[t2]= sorted(list(data_folder.glob('2021-05-*/' + t + '*')))
    
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
    
    type_df = all_types_to_df(files_by_type)
    
    for t in types2:
        d = type_df[t][0].columns.values
        print(type_df[t][0].columns)
        for i in type_df[t]:
            assert np.all(i.columns.values == d), (i,d)
            
    type_per_df = { t:pd.concat(type_df[t], ignore_index=True) for t in types2}
    df_final = reduce(lambda left,right: pd.merge(left,right,on='czas'), type_per_df.values())
    del df_final['man_unnamed: 5']
    del df_final['zcne_unnamed: 5']
    

    
    new_cols = [translate_naming.get(cname) or cname for cname in df_final.columns]
    df_final.columns = new_cols
    df_final = df_final.drop(['stra_1','stra_2','stra_3','stra_4','stra_5'], axis=1, errors='ignore')
    
    df_final['czas'] = pd.to_datetime(df_final['czas'])
    df_final = df_final.resample(f'{RESAMPLE_MIN}min', on='czas').mean()
    df_final['stra_sum'] = df_final['stra_sum'].rolling(RESAMPLE_MIN).mean().shift(-RESAMPLE_MIN)
    df_final = df_final[:-RESAMPLE_MIN]
    
    return df_final

def load_prediction_model(model_path: Path):
    model = pickle.load(open(model_path, "rb"))
    return model
    
model = load_prediction_model("model1.dat")

# df_final = load_sensor_data("data/01_raw")
df_final = load_sensor_data("./dane/")

X = df_final.iloc[:,:-1]

SETTINGS_CHANGE_UNIT = [80.0, 0.8, 2.0, 0.5]  # how hard it is to adjust param
ADJUSTMENT_TARGET_WEIGHT = 0.05 # how important is adjustment cost against target difference

def find_winning_params(state, current_settings, target):
    params = np.ones((300,4))  # params configuration which we want to try
    params[:,0] = np.linspace(1900, 3500, 300)
    params[:,1] = np.linspace(65, 81, 300)
    params[:,2] = np.linspace(40, 70, 300)
    params[:,3] = np.linspace(13, 27, 300)
    

    aaa = np.ones((params.shape[0], state.shape[1])) * state.T
    aab = np.concatenate([params, aaa], axis=1)

    predictions = model.predict(aab) # predict for all the params configurations

    target_diff = np.abs(predictions - target)
    adjustment_cost = np.sum(np.abs(aab[:,:4] - current_settings.reshape((1,4))) / SETTINGS_CHANGE_UNIT, axis=1)
    winner = np.argmin(target_diff + adjustment_cost * ADJUSTMENT_TARGET_WEIGHT)

    return params[winner,:] #settings which we believe are the best to obtain target from current 


def on_value_change(change):
    parameters = find_winning_params(state, current_settings, change)
    v1, v2, v3, v4 = map(lambda x : round(x, 2), parameters)
    st = f"""Przepływ powietrza: $${v1} [Nm^{3}/h]$$   
    Zawartość tlenu: $${v2} [\%]$$  
    Prędkość dmuchu $${v3} [M/s]$$  
    Nadawa pyłów procesowych $${v4} [Mg/h]$$"""
    return st

heat_loss = st.slider("Ustaw pożądaną stratę cieplną [MW]", min_value=13.37, max_value=21.37, step=0.5)


if st.button("Przesuń się o jedną minutę"):
    ss.i += 1
    st.write(ss.i)

state_df = X.iloc[ss.i,4:]
state = state_df.values.reshape((-1, 1))# real-time input from sensors
current_settings = X.iloc[ss.i,:4].values

reverse_translate_naming = {v:k for k, v in translate_naming.items()}
cols = [reverse_translate_naming[idx] for idx in state_df.index]

display = state_df.T
display.index = cols

st.markdown(on_value_change(heat_loss))
st.dataframe(display, width=860, height=640)
