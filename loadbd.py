import apps.database as DB
import apps.authentication.models as models

db = DB.get_session()

def insertConditions():
    data = [{"name": "Below","description":"Is below of"},
            {"name": "Above","description":"Is above of"},
            {"name": "Crosses Up","description":"Is crossing up"},
            {"name": "Crosses Down","description":"Is crossing down"},
            {"name": "Equal","description":"Is equal than"},
            {"name": "Major","description":"Is major than"},
            {"name": "Minor","description":"Is minor than"},
            {"name": "Distinct","description":"Is distinct of"},
            {"name": "Reached","description":"Is reached"}
            ]
    condition = [models.Condition(
        name = item["name"],
        description = item["description"]
    ) for item in data]
    # Add user to database
    db.add_all(condition)
    db.commit()

def insertPortfolioMetrics():
    data = [{"name": "Sharpe Ratio"},
            {"name": "Sortino Ratio"},
            {"name": "Jensen Alpha"},
            {"name": "ROI"},
            {"name": "Abs Drawdown"},
            {"name": "Max Drawdown"},
            {"name": "Rel Drawdown"},
            {"name": "Profit/Loss"},
            {"name": "% Profit"},
            {"name": "Losses"},
            {"name": "Wins"},
            {"name": "Trade Duration"},
            {"name": "N Buys"},
            {"name": "N BuysAcum"}
            ]
    metrics = [models.PortfolioMetric(
        name = item["name"]
    ) for item in data]
    # Add user to database
    db.add_all(metrics)
    db.commit()

def buyStrategies():
    data = [{"name": "DCA", "parameters": '{ "profit": 0.015 }', "description": "Dolar Cost Average. Periodic buys with same amount. Accumulative strategy"},
            {"name": "Incremental DCA", "parameters": '{ "profit": 0.015 }', "description": "Dolar Cost Average. Periodic buys with incremental amounts. Accumulative strategy"},
            {"name": "Incremental Buys", 
                "parameters": """{ 
                    "target": {
                        "profit": [0.015,0.01],"risk": [0.015,0.01],"risk_candles": [16,8],"profit_candles": [8,6]
                    }, 
                    "filter": {
                        "drop_rsi_above": [30,28]
                    },
                    "backtest": {
                        "initial_max": [[10,14],[15,20]],
                        "distance": [8,12],
                        "loss": [0],    
                        "max_capital_per_pair": 0.1
                    }  
                }""", 
                "description": "Purchase with incremental amounts in every buy signal until reach a fixed profit for all stacked quantity. SemiAccumulative"
            },
            {"name": "Scalping", "parameters": '{ "profit":0.015  }', "description": "Buy with a fixed amount. When profit or stoploss is reached sell inmediatly"},
            {"name": "Normal Trading", "parameters": '{ "profit": 0.015 }', "description": "Trade with a fixed amount based on buy and sell indicators signal. Can set stoploss"}
            ]
    buyStrategies = [models.BuyStrategy(
        name = item["name"],
        parameters = item["parameters"],
        description = item["description"]
    ) for item in data]
    # Add user to database
    db.add_all(buyStrategies)
    db.commit()

def insertPairs():
    data = [
    {"name": "BTCUSDT","exchange":"binance"},{"name": "ETHUSDT","exchange":"binance"},{"name": "SOLUSDT","exchange":"binance"},
    {"name": "DOTUSDT","exchange":"binance"},{"name": "LINKUSDT","exchange":"binance"},{"name": "BNBUSDT","exchange":"binance"},
    {"name": "ADAUSDT","exchange":"binance"},{"name": "CHZUSDT","exchange":"binance"},{"name": "XMRUSDT","exchange":"binance"},
    {"name": "ATOMUSDT","exchange":"binance"},{"name": "MATICUSDT","exchange":"binance"},{"name": "LTCUSDT","exchange":"binance"},
    {"name": "UNIUSDT","exchange":"binance"},{"name": "AVAXUSDT","exchange":"binance"},{"name": "SANDUSDT","exchange":"binance"},
    {"name": "NEARUSDT","exchange":"binance"},{"name": "FTMUSDT","exchange":"binance"},{"name": "ALGOUSDT","exchange":"binance"},
    {"name": "WAVESUSDT","exchange":"binance"},{"name": "MANAUSDT","exchange":"binance"},{"name": "ZECUSDT","exchange":"binance"},
    {"name": "MANAUSDT","exchange":"binance"},{"name": "CRVUSDT","exchange":"binance"},{"name": "AAVEUSDT","exchange":"binance"},
    {"name": "TFUELUSDT","exchange":"binance"},{"name": "ENSUSDT","exchange":"binance"},{"name": "IOTAUSDT","exchange":"binance"},
    {"name": "VETUSDT","exchange":"binance"},{"name": "RUNEUSDT","exchange":"binance"},{"name": "THETAUSDT","exchange":"binance"},
    {"name": "ENJUSDT","exchange":"binance"},{"name": "SNTUSDT","exchange":"binance"},{"name": "ZILUSDT","exchange":"binance"},
    {"name": "STXUSDT","exchange":"binance"},{"name": "MKRUSDT","exchange":"binance"},{"name": "KMDUSDT","exchange":"binance"},
    {"name": "BATUSDT","exchange":"binance"},{"name": "LPTUSDT","exchange":"binance"},{"name": "UMAUSDT","exchange":"binance"},
    {"name": "ETHBTC","exchange":"binance"},{"name": "SOLBTC","exchange":"binance"},
    {"name": "DOTBTC","exchange":"binance"},{"name": "LINKBTC","exchange":"binance"},{"name": "BNBBTC","exchange":"binance"},
    {"name": "ADABTC","exchange":"binance"},{"name": "CHZBTC","exchange":"binance"},{"name": "XMRBTC","exchange":"binance"},
    {"name": "ATOMBTC","exchange":"binance"},{"name": "MATICBTC","exchange":"binance"},{"name": "LTCBTC","exchange":"binance"},
    {"name": "UNIBTC","exchange":"binance"},{"name": "AVAXBTC","exchange":"binance"},{"name": "SANDBTC","exchange":"binance"},
    {"name": "NEARBTC","exchange":"binance"},{"name": "FTMBTC","exchange":"binance"},{"name": "ALGOBTC","exchange":"binance"},
    {"name": "WAVESBTC","exchange":"binance"},{"name": "MANABTC","exchange":"binance"},{"name": "ZECBTC","exchange":"binance"},
    {"name": "MANABTC","exchange":"binance"},{"name": "CRVBTC","exchange":"binance"},{"name": "AAVEBTC","exchange":"binance"},
    {"name": "TFUELBTC","exchange":"binance"},{"name": "ENSBTC","exchange":"binance"},{"name": "IOTABTC","exchange":"binance"},
    {"name": "VETBTC","exchange":"binance"},{"name": "RUNEBTC","exchange":"binance"},{"name": "THETABTC","exchange":"binance"},
    {"name": "ENJBTC","exchange":"binance"},{"name": "SNTBTC","exchange":"binance"},{"name": "ZILBTC","exchange":"binance"},
    {"name": "STXBTC","exchange":"binance"},{"name": "MKRBTC","exchange":"binance"},{"name": "KMDBTC","exchange":"binance"},
    {"name": "BATBTC","exchange":"binance"},{"name": "LPTBTC","exchange":"binance"},{"name": "UMABTC","exchange":"binance"}
            ]
    pairs = [models.Pair(
        name = item["name"],
        exchange = item["exchange"]
    ) for item in data]
    # Add user to database
    db.add_all(pairs)
    db.commit()

def insertIndicators():
    data = [{"name": "value","description":"Numeric Value"},{"name": "adx","description":"Average Directional Index"},{"name": "ao","description":"Awesome Oscillator"},
            {"name": "lowerband","description":"Lower Bollinger Band"},{"name": "upperband","description":"upper Bollinger Band"},
            {"name": "middleband","description":"Middle Bollinger Band"},{"name": "ema9","description":"9-Exponential Moving Average"},
            {"name": "ema13","description":"13-Exponential Moving Average"},{"name": "ema21","description":"21-Exponential Moving Average"},
            {"name": "ema50","description":"50-Exponential Moving Average"},{"name": "ema55","description":"55-Exponential Moving Average"},
            {"name": "ema100","description":"100-Exponential Moving Average"},{"name": "ema200","description":"200-Exponential Moving Average"},
            {"name": "ema400","description":"400-Exponential Moving Average"},{"name": "ma9","description":"9-Moving Average"},
            {"name": "ma13","description":"13-Moving Average"},{"name": "ma21","description":"21-Moving Average"},
            {"name": "ma50","description":"50-Moving Average"},{"name": "ma55","description":"55-Moving Average"},
            {"name": "ma100","description":"100-Moving Average"},{"name": "ma200","description":"200-Moving Average"},
            {"name": "ma400","description":"400-Moving Average"},{"name": "sma9","description":"9-Simple Moving Average"},
            {"name": "sma13","description":"13-Simple Moving Average"},{"name": "sma21","description":"21-Simple Moving Average"},
            {"name": "sma50","description":"50-Simple Moving Average"},{"name": "sma55","description":"55-Simple Moving Average"},
            {"name": "sma100","description":"100-Simple Moving Average"},{"name": "sma200","description":"200-Simple Moving Average"},
            {"name": "sma400","description":"400-Simple Moving Average"},
            {"name": "sar","description":"Parabolic SAR"},{"name": "cci","description":"Commodity Channel Index"},
            {"name": "macd","description":"Moving Average Convergence/Divergence"},{"name": "macd_signal","description":"Moving Average Convergence/Divergence"},
            {"name": "macd_hist","description":"Moving Average Convergence/Divergence"},{"name": "mfi","description":"Money Flow Index"},
            {"name": "mom","description":"Momentum"},{"name": "rsi","description":"Relative Strength Index"},
            {"name": "slowk","description":"Stochastic"},{"name": "slowd","description":"Stochastic"},
            {"name": "fastk","description":"Stochastic Relative Strength Index"},{"name": "fastd","description":"Stochastic Relative Strength Index"},
            {"name": "willr","description":"Williams' %R"},{"name": "obv","description":"On Balance Volume"},
            {"name": "open","description":"Open Price"},{"name": "close","description":"Close Price"},
            {"name": "high","description":"High Price"},{"name": "low","description":"Low Price"},
            {"name": "profit","description":"Profit Percent (between 0 and 1)"},{"name": "stoploss","description":"Stop Loss (between 0 and 1)"},
            {"name": "volume","description":"Volume"},{"name": "change","description":"Price Change"}]
    indicators = [models.Indicator(
        name = item["name"],
        description = item["description"]
    ) for item in data]
    # Add user to database
    db.add_all(indicators)
    db.commit()

def insertAlgorithm():
    data = [{"name": "RandomForest","parameters":"{'n_stimators': [800,1000], 'min_samples_split': [4],'cv': 5,'random_state': 0}","type":"ML"},
            {"name": "XGBoost","parameters":"{'learning_rate': [0.1],'objective': ['binary:logistic'],'cv': 5,'n_estimators': [1000,1200],'random_state': 0}","type":"ML"},
            {"name": "SVM","parameters":"{'kernel': ['rbf','linear'],'cv': 5,'random_state': 0}","type":"ML"},
            {"name": "KNN","parameters":"{'metric': ['euclidean','minkowski'],'cv': 5,'n_neighbors': [5,7]}","type":"ML"}]
    condition = [models.Algorithm(
        name = item["name"],
        parameters = item["parameters"],
        type = item["type"]
    ) for item in data]
    # Add user to database
    db.add_all(condition)
    db.commit()

def insertModelMetric():
    data = [{"name": "F1-score"},
            {"name": "ROC"},
            {"name": "Specificity"},
            {"name": "Accuracy"},
            {"name": "Recall"},
            {"name": "Precision"},
            {"name": "TP"},
            {"name": "TN"},
            {"name": "FP"},
            {"name": "FN"}
            ]
    condition = [models.ModelMetric(
        name = item["name"]
    ) for item in data]
    # Add user to database
    db.add_all(condition)
    db.commit()


insertConditions()
insertIndicators()
insertPairs()
insertPortfolioMetrics()
buyStrategies()
insertAlgorithm()
insertModelMetric()




