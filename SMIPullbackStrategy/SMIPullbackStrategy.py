from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import stoploss_from_open
from functools import reduce


class SMIPullbackStrategy(IStrategy):
    """
    基于您提供的四个代码片段的策略：
    1. 动态分段止损
    2. 回调检测
    3. SMI 趋势指标
    4. Kernel SMI
    """
    
    INTERFACE_VERSION = 3
    can_short = False
    
    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.02,
        "120": 0.015
    }
    
    stoploss = -0.10
    use_custom_stoploss = True
    trailing_stop = False
    
    timeframe = '5m'
    startup_candle_count: int = 50
    
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    # ==================== 片段1：动态止损参数 ====================
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.10, decimals=3, space='sell', optimize=True, load=True)
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=True, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=True, load=True)
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.070, decimals=3, space='sell', optimize=True, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.030, decimals=3, space='sell', optimize=True, load=True)
    
    # ==================== 片段2：回调检测参数 ====================
    pullback_periods = IntParameter(20, 40, default=30, space='buy', optimize=True, load=True)
    pullback_method = 'pct_outlier'  # 'stdev_outlier', 'pct_outlier', 'candle_body'
    
    # ==================== 片段3：SMI 趋势参数 ====================
    smi_k_length = IntParameter(5, 15, default=9, space='buy', optimize=True, load=True)
    smi_d_length = IntParameter(2, 5, default=3, space='buy', optimize=True, load=True)
    smi_smoothing = IntParameter(5, 20, default=10, space='buy', optimize=True, load=True)
    
    # ==================== 片段4：Kernel SMI 参数 ====================
    ksmi_K = IntParameter(5, 15, default=10, space='buy', optimize=True, load=True)
    ksmi_h = DecimalParameter(5.0, 15.0, default=8.0, decimals=1, space='buy', optimize=True, load=True)
    ksmi_rw = DecimalParameter(5.0, 15.0, default=8.0, decimals=1, space='buy', optimize=True, load=True)
    ksmi_x0 = IntParameter(3, 8, default=5, space='buy', optimize=True, load=True)
    ksmi_osint = IntParameter(30, 50, default=40, space='buy', optimize=True, load=True)
    ksmi_obint = IntParameter(30, 50, default=40, space='buy', optimize=True, load=True)
    
    # ==================== 片段2：回调检测 ====================
    def detect_pullback(self, df: DataFrame, periods=30, method='pct_outlier'):
        """回调和异常值检测"""
        if method == 'stdev_outlier':
            outlier_threshold = 2.0
            df['dif'] = df['close'] - df['close'].shift(1)
            df['dif_squared_sum'] = (df['dif']**2).rolling(window=periods + 1).sum()
            df['std'] = np.sqrt((df['dif_squared_sum'] - df['dif'].shift(0)**2) / (periods - 1))
            df['z'] = df['dif'] / df['std']
            df['pullback_flag'] = np.where(df['z'] >= outlier_threshold, 1, 0)
            df['pullback_flag'] = np.where(df['z'] <= -outlier_threshold, -1, df['pullback_flag'])
            
        elif method == 'pct_outlier':
            outlier_threshold = 2.0
            df["pb_pct_change"] = df["close"].pct_change()
            mean = df["pb_pct_change"].rolling(window=periods).mean()
            std = df["pb_pct_change"].rolling(window=periods).std()
            df['pb_zscore'] = (df["pb_pct_change"] - mean) / std
            df['pullback_flag'] = np.where(df['pb_zscore'] >= outlier_threshold, 1, 0)
            df['pullback_flag'] = np.where(df['pb_zscore'] <= -outlier_threshold, -1, df['pullback_flag'])
        
        elif method == 'candle_body':
            pullback_pct = 1.0
            df['change'] = df['close'] - df['open']
            df['pullback'] = (df['change'] / df['open']) * 100
            df['pullback_flag'] = np.where(df['pullback'] >= pullback_pct, 1, 0)
            df['pullback_flag'] = np.where(df['pullback'] <= -pullback_pct, -1, df['pullback_flag'])
        
        return df
    
    # ==================== 片段3：SMI 趋势指标 ====================
    def smi_trend(self, df: DataFrame, k_length=9, d_length=3, smoothing_type='EMA', smoothing=10):
        """Stochastic Momentum Index (SMI) 趋势指标"""
        ll = df['low'].rolling(window=k_length).min()
        hh = df['high'].rolling(window=k_length).max()
        diff = hh - ll
        rdiff = df['close'] - (hh + ll) / 2
        avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
        avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()
        smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
        
        smi_ma = ta.EMA(smi, timeperiod=smoothing)
        
        conditions = [
            (np.greater(smi, 0) & np.greater(smi, smi_ma)),  # (2) 强烈看涨
            (np.less(smi, 0) & np.greater(smi, smi_ma)),     # (1) 可能看涨反转
            (np.greater(smi, 0) & np.less(smi, smi_ma)),     # (-1) 可能看跌反转
            (np.less(smi, 0) & np.less(smi, smi_ma))         # (-2) 强烈看跌
        ]
        smi_trend = np.select(conditions, [2, 1, -1, -2])
        
        return smi, smi_ma, smi_trend
    
    # ==================== 片段4：Kernel SMI ====================
    def calculate_smi_kernel(self, df: DataFrame, _K: int = 10, h: float = 8.0, 
                            rw: float = 8.0, x_0: int = 5, osint: int = 40, 
                            obint: int = 40, _col: str = 'close'):
        """Kernel 回归 SMI 计算"""
        def kernel_regression(_src):
            if len(_src) < x_0:
                return 0
            weights = [(1 + (i**2 / (h**2 * 2 * rw)))**(-rw) for i in range(len(_src))]
            weighted_sum = sum([val*weight for val, weight in zip(_src[x_0:], weights)])
            return weighted_sum / sum(weights) if sum(weights) > 0 else 0
        
        df['highestHigh'] = df[_col].rolling(window=_K).max()
        df['lowestLow'] = df[_col].rolling(window=_K).min()
        df['highestLowestRange'] = df['highestHigh'] - df['lowestLow']
        df['relativeRange'] = df[_col] - (df['highestHigh'] + df['lowestLow']) / 2
        df['smi_k'] = 200 * (
            df['relativeRange'].rolling(window=_K).apply(kernel_regression, raw=True) /
            df['highestLowestRange'].rolling(window=_K).apply(kernel_regression, raw=True).replace(0, 1)
        )
        df['k_smi'] = df['smi_k'].rolling(window=_K).apply(kernel_regression, raw=True)
        
        df['k_smi_down'] = (df['smi_k'] < obint) & (df['smi_k'].shift(1) >= obint)
        df['k_smi_up'] = (df['smi_k'] > -osint) & (df['smi_k'].shift(1) <= -osint)
        
        return df['smi_k'], df['k_smi'], df['k_smi_down'], df['k_smi_up']
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """使用四个片段的所有指标"""
        
        # 片段2：回调检测
        dataframe = self.detect_pullback(
            dataframe, 
            periods=self.pullback_periods.value,
            method=self.pullback_method
        )
        
        # 片段3：SMI 趋势指标
        dataframe['smi'], dataframe['smi_ma'], dataframe['smi_trend'] = self.smi_trend(
            dataframe,
            k_length=self.smi_k_length.value,
            d_length=self.smi_d_length.value,
            smoothing_type='EMA',
            smoothing=self.smi_smoothing.value
        )
        
        # 片段4：Kernel SMI
        dataframe['smi_k'], dataframe['k_smi'], dataframe['k_smi_down'], dataframe['k_smi_up'] = self.calculate_smi_kernel(
            dataframe,
            _K=self.ksmi_K.value,
            h=self.ksmi_h.value,
            rw=self.ksmi_rw.value,
            x_0=self.ksmi_x0.value,
            osint=self.ksmi_osint.value,
            obint=self.ksmi_obint.value
        )
        
        # 辅助指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        买入逻辑：结合四个片段
        
        1. Kernel SMI 上穿超卖区（片段4）
        2. SMI 趋势看涨（片段3）
        3. 下跌回调后反弹（片段2）
        """
        conditions = []
        
        # 片段4：Kernel SMI 上穿超卖区
        conditions.append(dataframe['k_smi_up'] == True)
        
        # 片段3：SMI 趋势至少是可能看涨
        conditions.append(dataframe['smi_trend'] >= 1)
        
        # 片段2：检测到下跌回调（前1-3根蜡烛）
        conditions.append(
            (dataframe['pullback_flag'].shift(1) == -1) |
            (dataframe['pullback_flag'].shift(2) == -1) |
            (dataframe['pullback_flag'].shift(3) == -1)
        )
        
        # 辅助条件：RSI 不超买
        conditions.append(dataframe['rsi'] < 70)
        
        # 成交量确认
        conditions.append(dataframe['volume'] > dataframe['volume_mean'] * 0.5)
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        卖出逻辑：结合四个片段
        
        1. Kernel SMI 下穿超买区（片段4）
        2. SMI 趋势看跌（片段3）
        3. 上涨异常（片段2）
        """
        conditions = []
        
        # 片段4：Kernel SMI 下穿超买区
        conditions.append(dataframe['k_smi_down'] == True)
        
        # 片段3：SMI 趋势看跌
        conditions.append(dataframe['smi_trend'] <= -1)
        
        # 片段2：检测到上涨异常
        conditions.append(dataframe['pullback_flag'] == 1)
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        
        return dataframe
    
    # ==================== 片段1：动态止损 ====================
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        片段1的原始动态止损逻辑
        """
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value
        
        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL
        
        if (sl_profit >= current_profit):
            return -0.99
        
        return stoploss_from_open(sl_profit, current_profit)