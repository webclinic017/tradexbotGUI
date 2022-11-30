# -*- encoding: utf-8 -*-

# models.py

from ..database import Base
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, DateTime, FetchedValue, Boolean, ForeignKey, Float
from sqlalchemy_serializer import SerializerMixin
from fastapi_utils.guid_type import GUID, GUID_DEFAULT_SQLITE

class Algorithm(Base, SerializerMixin):
    serialize_only={'name'}
    __tablename__ = "algorithm"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    parameters = Column(String(2000))
    type = Column(String(200))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    algorithm_details = relationship("AlgorithmDetail", back_populates="algorithm", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Bot(Base, SerializerMixin):
    serialize_only={'name','uuid','train_model','task_id'}
    __tablename__ = "bot"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    train_model_id = Column(Integer, ForeignKey("train_model.id"))
    uuid = Column(String(50))
    status = Column(String(50)) # running, stopped
    task_id = Column(String(100))

    active = Column(Boolean, default=True)

    # Relation with childrens Has many

    # Relation with parents Has one
    train_model = relationship("TrainModel", back_populates="bots")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class AlgorithmDetail(Base, SerializerMixin):
    serialize_only={'parameters','algorithm.name'}
    __tablename__ = "algorithm_detail"
    #normal columns
    id = Column(Integer, primary_key=True)
    training_detail_id = Column(Integer, ForeignKey("training_detail.id"))
    algorithm_id = Column(Integer, ForeignKey("algorithm.id"))
    name = Column(String(200))
    parameters = Column(String(3000))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    # Relation with parents Has one
    training_detail = relationship("TrainingDetail", back_populates="algorithm_details")
    algorithm = relationship("Algorithm", back_populates="algorithm_details")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())
class Backtest(Base, SerializerMixin):

    __tablename__ = "backtest"
    #normal columns
    id = Column(Integer, primary_key=True)
    #strategy_id = Column(Integer, ForeignKey("strategy.id"))
    train_model_id = Column(Integer, ForeignKey("train_model.id"))
    pair = Column(String(10))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    backtest_charts = relationship("BacktestChart", back_populates="backtest", cascade = "all, delete, delete-orphan")
    backtest_metrics = relationship("BacktestMetric", back_populates="backtest", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one
    #training_detail = relationship("TrainingDetail", back_populates="backtests")
    #strategy = relationship("Strategy", back_populates="backtests")
    train_model = relationship("TrainModel", back_populates="backtests")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class BacktestChart(Base, SerializerMixin):

    __tablename__ = "backtest_chart"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    path = Column(String(200))
    backtest_id = Column(Integer, ForeignKey("backtest.id"))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many

    # Relation with parents Has one
    backtest = relationship("Backtest", back_populates="backtest_charts")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class BacktestMetric(Base, SerializerMixin):
    __tablename__ = "backtest_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    value = Column(Float)
    description = Column(String(200))
    backtest_id = Column(Integer, ForeignKey("backtest.id"))
    portfolio_metric_id = Column(Integer, ForeignKey("portfolio_metric.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    backtest = relationship("Backtest", back_populates="backtest_metrics")
    portfolio_metric = relationship("PortfolioMetric", back_populates="backtest_metrics")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class BuyStrategy(Base, SerializerMixin):
    serialize_only={'name'}
    __tablename__ = "buy_strategy"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    parameters = Column(String(3000))
    description = Column(String(1000))
    active = Column(Boolean, default=True)
    
    # Relation with childrens Has many
    strategies = relationship("Strategy", back_populates="buy_strategy", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Condition(Base, SerializerMixin):
    __tablename__ = "condition"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(500))
    active = Column(Boolean, default=True)
    
    # Relation with childrens Has many

    # NO BIDIRECCIONAL

    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Indicator(Base, SerializerMixin):
    serialize_only={'name'}
    __tablename__ = "indicator"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(20))
    description = Column(String(200))
    params = Column(String(500))
    active = Column(Boolean, default=True)
    
    # Relation with childrens Has many
    train_indicators = relationship("TrainIndicator", back_populates="indicator", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class ModelMetric(Base, SerializerMixin):
    serialize_only={'name'}
    __tablename__ = "model_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    training_metrics = relationship("TrainingMetric", back_populates="model_metric", cascade = "all, delete, delete-orphan")
    train_metrics = relationship("TrainMetric", back_populates="model_metric", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class PortfolioMetric(Base, SerializerMixin):
    serialize_only={'name'}
    __tablename__ = "portfolio_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    strategy_metrics = relationship("StrategyMetric", back_populates="portfolio_metric", cascade = "all, delete, delete-orphan")
    backtest_metrics = relationship("BacktestMetric", back_populates="portfolio_metric", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Pair(Base, SerializerMixin):
    serialize_only={'name'}
    __tablename__ = "pair"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(20))
    exchange = Column(String(100))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    strategy_pairs = relationship("StrategyPair", back_populates="pair", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Rule(Base, SerializerMixin):
    serialize_only={'order_type','first_indicator.name','second_indicator.name','condition.name','value'}
    __tablename__ = "rule"
    #normal columns
    id = Column(Integer, primary_key=True)
    order_type = Column(Boolean, default=True) # True == Buy False == Sell
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    first_indicator_id = Column(Integer, ForeignKey("indicator.id"))
    second_indicator_id = Column(Integer, ForeignKey("indicator.id"))
    condition_id = Column(Integer, ForeignKey("condition.id"))
    value = Column(Float)
    # Relation with childrens Has many

    # Relation with parents Has one
    strategy = relationship("Strategy", back_populates="rules")
    first_indicator = relationship("Indicator", foreign_keys='Rule.first_indicator_id')
    second_indicator = relationship("Indicator", foreign_keys='Rule.second_indicator_id')
    condition = relationship("Condition")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

"""
class SellStrategy(Base, SerializerMixin):

    __tablename__ = "sell_strategy"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    active = Column(Boolean, default=True)
    
    # Relation with childrens Has many
    strategies = relationship("Strategy", back_populates="sell_strategy")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())
"""
class Strategy(Base, SerializerMixin):
    serialize_only={'strategy_parameters','buy_strategy.name','rules','strategy_pairs','strategy_metrics'}
    __tablename__ = "strategy"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    strategy_parameters = Column(String(3000))
    json_parameters = Column(String(3000))
    buy_strategy_id = Column(Integer, ForeignKey("buy_strategy.id"))
    user_id = Column(Integer, ForeignKey("user.id"))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    rules = relationship("Rule", back_populates="strategy", cascade = "all, delete, delete-orphan")
    strategy_pairs = relationship("StrategyPair", back_populates="strategy", cascade = "all, delete, delete-orphan")
    training_settings = relationship("TrainSetup", back_populates="strategy", cascade = "all, delete, delete-orphan" )
    trades = relationship("Trade", back_populates="strategy", cascade = "all, delete, delete-orphan")
    strategy_metrics = relationship("StrategyMetric", back_populates="strategy", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one
    buy_strategy = relationship("BuyStrategy", back_populates="strategies")
    user = relationship("User", back_populates="strategies")

    created_at = Column(DateTime, default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime, onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class StrategyMetric(Base, SerializerMixin):
    serialize_only={'portfolio_metric.name'}
    __tablename__ = "strategy_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    value = Column(Float)
    description = Column(String(200))
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    portfolio_metric_id = Column(Integer, ForeignKey("portfolio_metric.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    strategy = relationship("Strategy", back_populates="strategy_metrics")
    portfolio_metric = relationship("PortfolioMetric", back_populates="strategy_metrics")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class StrategyPair(Base, SerializerMixin):
    serialize_only={'pair.name'}
    __tablename__ = "strategy_pair"
    #normal columns
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    pair_id = Column(Integer, ForeignKey("pair.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    strategy = relationship("Strategy", back_populates="strategy_pairs")
    pair = relationship("Pair", back_populates="strategy_pairs")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Trade(Base, SerializerMixin):
    __tablename__ = "trade"
    #normal columns
    id = Column(Integer, primary_key=True)
    trading_setup_id = Column(Integer, ForeignKey("trading_setup.id"))
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    quantity = Column(Float)
    buy_amount = Column(Float)
    sell_amount = Column(Float)
    entry_price = Column(Float)
    sell_price = Column(Float)
    stop_price = Column(Float)
    duration = Column(Integer) # cuantas velas demoró en cerrarse
    state = Column(Integer)  # Abierto 1, Cerrado 0, Stop -1
    # Relation with childrens Has many

    # Relation with parents Has one
    trading_setup = relationship("TradingSetup", back_populates="trades")
    strategy = relationship("Strategy", back_populates="trades")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TradingSetup(Base, SerializerMixin):
    serialize_only = ('id', 'exchange','exchange_comission', 'capital', 'currency_base', 'api_key','api_secret')

    __tablename__ = "trading_setup"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    active = Column(Boolean, default=True)
    exchange = Column(String(2000))
    exchange_comission = Column(Float(precision=4))
    capital = Column(Float(precision=8))
    currency_base = Column(String(10))
    currency_symbol = Column(String(5))
    api_key = Column(String(100))
    api_secret = Column(String(100))
        
    user_id = Column(Integer, ForeignKey("user.id"))

    # Relation with childrens Has many
    trades = relationship("Trade", back_populates="trading_setup", cascade = "all, delete, delete-orphan")
    training_settings = relationship("TrainSetup", back_populates="trading_setup", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one
    user = relationship("User", back_populates="trading_settings")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Train(Base, SerializerMixin):

    __tablename__ = "train"
    #normal columns
    id = Column(Integer, primary_key=True)
    train_setup_id = Column(Integer, ForeignKey("train_setup.id"))
    status = Column(String(100))
    completed = Column(Boolean,default=False)
    backtest_parameters = Column(String(2000))
    filter_parameters = Column(String(2000))
    target_parameters = Column(String(2000))
    checksum = Column(String(100))
    parameters = Column(String(8000))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    #backtests = relationship("Backtest", back_populates="train")
    train_models = relationship("TrainModel", back_populates="train", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one
    train_setup = relationship("TrainSetup", back_populates="trainings")

    created_at = Column(DateTime, default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime, onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())
class TrainModel(Base, SerializerMixin):

    __tablename__ = "train_model"
    #normal columns
    id = Column(Integer, primary_key=True)
    train_id = Column(Integer, ForeignKey("train.id"))
    model = Column(String(200))
    scaler = Column(String(200))
    columns = Column(String(1000))
    model_best_parameters = Column(String(2000))
    active = Column(Boolean, default=True)
    favorite = Column(Boolean, default=False)

    # Relation with childrens Has many
    backtests = relationship("Backtest", back_populates="train_model", cascade = "all, delete, delete-orphan")
    bots = relationship("Bot", back_populates="train_model", cascade = "all, delete, delete-orphan")
    train_metrics = relationship("TrainMetric", back_populates="train_model", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one
    train = relationship("Train", back_populates="train_models")

    created_at = Column(DateTime, default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime, onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())
    
    def get_metrics_value(self):
        ret = [item.value for item in self.train_metrics]
        return ret
class TrainIndicator(Base, SerializerMixin):
    serialize_only = {'indicator.name'}
    
    __tablename__ = "train_indicator"
    #normal columns
    id = Column(Integer, primary_key=True)
    train_setup_id = Column(Integer, ForeignKey("train_setup.id"))
    indicator_id = Column(Integer, ForeignKey("indicator.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    train_setup = relationship("TrainSetup", back_populates="train_indicators")
    indicator = relationship("Indicator", back_populates="train_indicators")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TrainingDetail(Base, SerializerMixin):
    serialize_only={'data_split','algorithm_details','training_metrics'}
    __tablename__ = "training_detail"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    data_split = Column(Float(precision=2))
    user_id = Column(Integer, ForeignKey("user.id"))
    #scaler = Column(String(200))
    #model = Column(String(200))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    training_settings = relationship("TrainSetup", back_populates="training_detail", cascade = "all, delete, delete-orphan")
    algorithm_details = relationship("AlgorithmDetail", back_populates="training_detail", cascade = "all, delete, delete-orphan")
    training_metrics = relationship("TrainingMetric", back_populates="training_detail", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one
    user = relationship("User", back_populates="training_details")

    created_at = Column(DateTime, default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime, onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TrainMetric(Base, SerializerMixin):
    __tablename__ = "train_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    value = Column(Float)
    description = Column(String(200))
    train_model_id = Column(Integer, ForeignKey("train_model.id"))
    model_metric_id = Column(Integer, ForeignKey("model_metric.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    train_model = relationship("TrainModel", back_populates="train_metrics")
    model_metric = relationship("ModelMetric", back_populates="train_metrics")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TrainingMetric(Base, SerializerMixin):
    serialize_only={'model_metric.name'}
    __tablename__ = "training_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    value = Column(Float)
    description = Column(String(200))
    training_detail_id = Column(Integer, ForeignKey("training_detail.id"))
    model_metric_id = Column(Integer, ForeignKey("model_metric.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    training_detail = relationship("TrainingDetail", back_populates="training_metrics")
    model_metric = relationship("ModelMetric", back_populates="training_metrics")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TrainSetup(Base, SerializerMixin):
    serialize_only = ('id','timeframe', 'train_data_start', 'train_data_end', 'backtest_data_start', 'backtest_data_end','train_indicators','training_detail','strategy','trading_setup')
    __tablename__ = "train_setup"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    active = Column(Boolean, default=True)
    timeframe = Column(String(10))
    train_data_start = Column(String(12))
    train_data_end = Column(String(12))
    backtest_data_start = Column(String(12))
    backtest_data_end = Column(String(12))

    user_id = Column(Integer, ForeignKey("user.id"))
    trading_setup_id = Column(Integer, ForeignKey("trading_setup.id"))
    training_detail_id = Column(Integer, ForeignKey("training_detail.id"))
    strategy_id = Column(Integer, ForeignKey("strategy.id"))

    # Relation with childrens Has many
    trainings = relationship("Train", back_populates="train_setup", cascade = "all, delete, delete-orphan")
    train_indicators = relationship("TrainIndicator", back_populates="train_setup", cascade = "all, delete, delete-orphan")
    # Relation with parents Has one
    user = relationship("User", back_populates="training_settings")
    trading_setup = relationship("TradingSetup", back_populates="training_settings")
    training_detail = relationship("TrainingDetail", back_populates="training_settings")
    strategy = relationship("Strategy", back_populates="training_settings")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class User(Base, SerializerMixin):

    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True)
    email = Column(String(128), unique=True)
    uuid = Column(GUID, default=GUID_DEFAULT_SQLITE)
    password = Column(String(64))
    strategies = relationship("Strategy", back_populates="user")
    training_details = relationship("TrainingDetail", back_populates="user", cascade = "all, delete, delete-orphan")
    training_settings = relationship("TrainSetup", back_populates="user", cascade = "all, delete, delete-orphan")
    trading_settings = relationship("TradingSetup", back_populates="user", cascade = "all, delete, delete-orphan")
    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())