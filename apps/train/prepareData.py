# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ..train.helpers import extract_indicators
import requests
from datetime import datetime

class PrepareData:
    """
    Clase con la cual se agregarán indicadores al dataset del tick dado.
    además se agregara la columna target. Esta clase devolverá el dataset listo para el entrenamiento
    """

    target = []
    dataframe = None
    tick = ""
    timeframe = "5m"
    start = "01-01-2017"
    end = "31-12-2022"
    split = "01-11-2021"
    target = {}
    x_columns = ["open_time","close_time", "high", "close", "open", "low","volume"]
    train_columns = ["volume"]
    train_dataset = pd.DataFrame()
    backtest_dataset = []
    headers = {}

    def __init__(self, configuration, api):
        print(configuration["train_indicators"])
        print(configuration["strategy"]["rules"])
        for item in extract_indicators(configuration["train_indicators"],configuration["strategy"]["rules"]):
            self.x_columns.append(item)
            self.train_columns.append(item)
        self.pairs = [item["pair"]["name"] for item in configuration["strategy"]["strategy_pairs"]]
        self.start = configuration["train_data_start"]
        self.end = configuration["backtest_data_end"]
        self.split = configuration["backtest_data_start"]
        self.timeframe = configuration["timeframe"]
        self.api = api
        self.loadDataset()
        
    
    def loadDataset(self):
        for pair in self.pairs:
            data = {
                "pair": pair,
                "timeframe": self.timeframe,
                "ini": self.start,
                "end": self.end,
                "indicators": self.x_columns
            }
            print(data)
            print(self.api["host"]+'/api/data')
            r = requests.post(self.api["host"]+'/api/data', json=data, headers=self.api["headers"])
            #print(r)
            #print(r.json()["data"])
            print("$$$$$$$$$$")
            df = pd.read_json(r.json()["data"])
            #print(df)
            df = df.set_index("open_time")
            df = df.dropna()
            self.train_dataset = pd.concat([self.train_dataset, df.loc[self.dateconvert(self.start) : self.dateconvert(self.split)]], axis=0)
            aux_df = df.loc[self.dateconvert(self.split) : self.dateconvert(self.end)].reset_index()
            self.backtest_dataset.append({"pair":pair, "data": aux_df})
        self.train_dataset = self.train_dataset.reset_index()
        print("######## SUCCESSFUL DATASET LOADER ##########")
    def dateconvert(self,date):
        aux = date.split("/")
        return aux[2]+"-"+aux[1]+"-"+aux[0]
    def showGraph(self):
        print(f"### Imprimiendo grafico del dataset")
        dataset1 = self.dataframe.loc[self.graph_start : self.graph_end]
        dataset1 = dataset1.reset_index()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(
            go.Candlestick(
                x=dataset1["open_time"],
                open=dataset1["open"],
                high=dataset1["high"],
                low=dataset1["low"],
                close=dataset1["close"],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=dataset1["open_time"], y=dataset1["rsi"]), row=2, col=1
        )
        win = 0
        for i in range(len(dataset1["resultado"])):
            temp = dataset1.loc[i:i, "resultado"].values[0]
            # temp1 = dataset1.loc[i:i,'Rsi'].values[0]
            if temp == True:
                fig.add_annotation(
                    x=dataset1.loc[i, "open_time"],
                    y=dataset1.loc[i, "close"],
                    text="▲",
                    showarrow=False,
                    font=dict(size=16, color="blue"),
                )
                win += 1
            # if temp1<target['min_rsi']:
            #    fig.add_annotation(x=dataset1.loc[i,'Open time'], y=dataset1.loc[i,'Close'], text="▲", showarrow=False, font=dict(size=12, color='green'))
        fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
        fig.write_html(f"./data/graphs/grafico_data_{self.tick}.html")
        fig.show()


