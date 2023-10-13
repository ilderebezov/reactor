import json
import os
import pickle
import random

import pandas as pd
from pandas import DataFrame
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.graph import plot


def create_update_model():
    """
    create or update predict model, based on data from "sample_data.csv"
    :return:
    """
    filename_data = 'sample_data.csv'
    check_file = os.path.isfile(filename_data)
    img = None
    if check_file:
        df_init = pd.read_csv(filepath_or_buffer=filename_data, index_col=False)
        sets_numbers = 8
        dfs = [df_init.iloc[i::8, :].reset_index(drop=True) for i in range(sets_numbers)]
        rnd_lst = random.sample(range(8), 8)
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        test_number = 0
        for number, value in enumerate(rnd_lst):
            if number > 6:
                test_number = value
                test_df = pd.concat([test_df, dfs[value]], ignore_index=True)
            else:
                train_df = pd.concat([train_df, dfs[value]], ignore_index=True)
        x_data = train_df[['t', 'B_flowrate']].values
        y_data = train_df['A'].values
        n_estimators = 1
        r2 = 0.0
        rnd_fr_model = None
        df_predict = None
        while r2 < 0.999:
            rnd_fr_model = RandomForestRegressor(n_estimators=10, max_features=2)
            rnd_fr_model.fit(x_data, y_data)
            df_predict = pd.DataFrame({'t': test_df['t'].values,
                                       'a': rnd_fr_model.predict(test_df[['t', 'B_flowrate']].values)})
            # calc R2 param
            r2 = r2_score(dfs[test_number]['A'].values, df_predict['a'].values)
            n_estimators += 1
        filename_model = 'rnd_reg_model.sav'
        with open(filename_model, 'wb') as file:
            pickle.dump(rnd_fr_model, file)
        img = plot(df_a=dfs[test_number], df_b=df_predict, r2=r2, graph_name="Model quality", file=False)
    return check_file, img


def predict_data(data_in: json = None) -> dict:
    if data_in is None:
        data_in = {"tspan": [0.0, 200.0],
                   "flowrate": {"0.0": 1.0}
                   }
        # data_in = {"tspan": [0.0, 200.0],
        #           "flowrate": {'0.0': 1.0, '3.0': 2.0, '5.0': 1.1}}
    filename_data = "sample_data.csv"
    df_init = pd.read_csv(filepath_or_buffer=filename_data, index_col=False)
    df_range = df_init[(df_init["t"] >= data_in["tspan"][0]) & (df_init['t'] <= data_in['tspan'][1])]
    filename_model = 'rnd_reg_model.sav'
    with open(filename_model, 'rb') as file:
        rnd_regress_model_from_pickle = pickle.load(file)
    time_line_r2 = df_init['t']
    df_predict_r2 = predict_point(time_line=time_line_r2, data_init=data_in, model=rnd_regress_model_from_pickle)
    time_line = [float(time) for time in range(int(data_in['tspan'][0]), int(data_in['tspan'][1]) + 1, 1)]
    if len(data_in["flowrate"]) == 1 and data_in["flowrate"].get("0.0") == 1.0:
        df_predict = pd.DataFrame({'t': time_line})
        df_predict["b_flowrate"] = data_in["flowrate"]["0.0"]
        df_predict["a"] = predict_point(time_line=time_line,
                                        data_init=data_in,
                                        model=rnd_regress_model_from_pickle)["a"].iloc[0]
    else:
        df_predict = predict_point(time_line=time_line, data_init=data_in, model=rnd_regress_model_from_pickle)

    r2 = r2_score(df_range['A'].values, df_predict_r2["a"].values)
    plot(df_a=df_range, df_b=df_predict, r2=r2, graph_name="Calc based on user data", file=True)
    return df_predict.to_dict('list')


def predict_point(time_line: list, data_init: dict, model: ensemble) -> DataFrame:
    """
    Do prediction based on input data
    :param time_line: time line
    :param data_init: input data for predict
    :param model: predict model
    :return: predict data
    """
    predicted_from_pickle = []
    b_flowrate = 1.0
    b_flowrate_lst = []
    for time in time_line:
        if data_init["flowrate"].get(f'{time}') is not None:
            b_flowrate = data_init["flowrate"][f'{time}']
        # Use the loaded pickled model to make predictions
        predicted_from_pickle.append(model.predict([[time, b_flowrate]])[0])
        b_flowrate_lst.append(b_flowrate)
    df_predict = pd.DataFrame({"t": time_line,
                               "b_flowrate": b_flowrate_lst,
                               "a": predicted_from_pickle})
    return df_predict
