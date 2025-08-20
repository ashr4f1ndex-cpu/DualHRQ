"""
public_data_loaders.py
======================

Comprehensive public data loaders for enhanced HRM training.
Integrates free and low-cost data sources to maximize training data
without expensive subscriptions.

Features:
- Yahoo Finance (completely free)
- Alpha Vantage (free tier: 25 calls/day)
- IEX Cloud (free tier: 50k credits/month)
- FRED (Federal Reserve Economic Data - free)
- VIX and volatility indices (CBOE free)
- Alternative data sources (sentiment, news)
- Crypto options (Deribit free API)
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from typing import Optional
import time
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ================ FREE DATA SOURCES ================

def load_yahoo_finance_options(symbol: str, start_date: str, end_date: str, 
                              target_dte: int = 30) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load options data using Yahoo Finance (completely free).
    
    Note: Yahoo Finance options data is limited but includes IV estimates.
    This loader reconstructs historical options series using current option chains
    and historical underlying prices.
    
    Args:
        symbol: Underlying symbol (e.g., 'SPY', 'AAPL')
        start_date: Start date for historical data
        end_date: End date for historical data
        target_dte: Target days to expiration
    
    Returns:
        Tuple of (S, iv_entry, iv_exit, expiry) Series
    """
    try:
        # Get historical underlying prices
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        S = hist['Close'].dropna()
        
        # Get current options chain to estimate historical IV structure
        options_dates = ticker.options
        if not options_dates:
            raise ValueError(f"No options available for {symbol}")
        
        # Find expiration closest to target DTE from today
        today = pd.Timestamp.now().normalize()
        target_exp = None
        min_dte_diff = float('inf')
        
        for exp_str in options_dates:
            exp_date = pd.to_datetime(exp_str)
            dte = (exp_date - today).days
            if abs(dte - target_dte) < min_dte_diff:
                min_dte_diff = abs(dte - target_dte)
                target_exp = exp_str
        
        if target_exp:
            # Get current option chain
            opt_chain = ticker.option_chain(target_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Find ATM options
            current_price = S.iloc[-1]
            calls['abs_moneyness'] = abs(calls['strike'] - current_price)
            puts['abs_moneyness'] = abs(puts['strike'] - current_price)
            
            atm_call = calls.loc[calls['abs_moneyness'].idxmin()]
            atm_put = puts.loc[puts['abs_moneyness'].idxmin()]
            
            # Use average IV as proxy for historical IV
            current_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2
            
            # Create synthetic IV series (constant IV assumption)
            iv_entry = pd.Series(current_iv, index=S.index, name='iv_entry')
            iv_exit = iv_entry.shift(-5).fillna(current_iv)  # 5-day forward proxy
            
            # Create expiry series
            expiry_date = pd.to_datetime(target_exp)
            expiry = pd.Series(expiry_date, index=S.index, name='expiry')
            
            return S, iv_entry, iv_exit, expiry
        else:
            raise ValueError(f"No suitable expiration found for {symbol}")
            
    except Exception as e:
        logger.error(f"Yahoo Finance loader failed for {symbol}: {e}")
        raise

def load_alpha_vantage_data(symbol: str, api_key: str, start_date: str, end_date: str,
                           target_dte: int = 30) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load data using Alpha Vantage API (free tier: 25 calls/day).
    
    Combines historical prices with volatility indicators to estimate IV.
    Uses GARCH model to estimate implied volatility proxy.
    
    Args:
        symbol: Stock symbol
        api_key: Alpha Vantage API key (free from alphavantage.co)
        start_date: Start date
        end_date: End date
        target_dte: Target DTE
    
    Returns:
        Tuple of (S, iv_entry, iv_exit, expiry) Series
    """
    base_url = "https://www.alphavantage.co/query"
    
    # Get daily adjusted prices
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': api_key
    }
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    if 'Error Message' in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    
    # Parse price data
    time_series = data['Time Series (Daily)']
    prices = []
    
    for date_str, values in time_series.items():
        date = pd.to_datetime(date_str)
        if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
            prices.append({
                'date': date,
                'close': float(values['5. adjusted close']),
                'volume': float(values['6. volume'])
            })
    
    df = pd.DataFrame(prices).set_index('date').sort_index()
    S = df['close']
    
    # Calculate returns and estimate IV using GARCH-like approach
    returns = np.log(S / S.shift(1)).dropna()
    
    # Simple EWMA volatility estimation (proxy for IV)
    lambda_param = 0.94
    var_ewma = returns.ewm(alpha=1-lambda_param).var()
    vol_ewma = np.sqrt(var_ewma * 252)  # Annualized
    
    # Align volatility with price series
    iv_entry = vol_ewma.reindex(S.index).fillna(method='ffill')
    iv_exit = iv_entry.shift(-5).fillna(method='ffill')
    
    # Create synthetic expiry (target_dte from each date)
    expiry = S.index + pd.Timedelta(days=target_dte)
    expiry = pd.Series(expiry, index=S.index, name='expiry')
    
    return S, iv_entry, iv_exit, expiry

def load_iex_cloud_data(symbol: str, token: str, start_date: str, end_date: str,
                       target_dte: int = 30) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load data using IEX Cloud (free tier: 50k credits/month).
    
    Provides high-quality price data and some volatility metrics.
    
    Args:
        symbol: Stock symbol
        token: IEX Cloud token (free from iexcloud.io)
        start_date: Start date
        end_date: End date
        target_dte: Target DTE
    
    Returns:
        Tuple of (S, iv_entry, iv_exit, expiry) Series
    """
    base_url = "https://cloud.iexapis.com/stable"
    
    # Calculate date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    range_str = f"{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
    
    # Get historical prices
    url = f"{base_url}/stock/{symbol}/chart/date/{range_str}"
    params = {'token': token}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"IEX Cloud API error: {response.status_code}")
    
    data = response.json()
    if not data:
        raise ValueError(f"No data returned for {symbol}")
    
    # Parse price data
    prices = []
    for day_data in data:
        prices.append({
            'date': pd.to_datetime(day_data['date']),
            'close': day_data['close'],
            'volume': day_data['volume']
        })
    
    df = pd.DataFrame(prices).set_index('date').sort_index()
    S = df['close']
    
    # Get additional volatility data if available
    try:
        vol_url = f"{base_url}/stock/{symbol}/stats"
        vol_response = requests.get(vol_url, params=params)
        vol_data = vol_response.json()
        
        # Use beta and other metrics to estimate volatility
        beta = vol_data.get('beta', 1.0)
        market_vol = 0.16  # Assume 16% market volatility
        estimated_vol = abs(beta) * market_vol
        
    except:
        # Fallback to historical volatility calculation
        returns = np.log(S / S.shift(1)).dropna()
        estimated_vol = returns.std() * np.sqrt(252)
    
    # Create IV series
    iv_entry = pd.Series(estimated_vol, index=S.index, name='iv_entry')
    iv_exit = iv_entry.shift(-5).fillna(estimated_vol)
    
    # Create expiry series
    expiry = S.index + pd.Timedelta(days=target_dte)
    expiry = pd.Series(expiry, index=S.index, name='expiry')
    
    return S, iv_entry, iv_exit, expiry

def load_fred_economic_data(series_id: str, start_date: str, end_date: str) -> pd.Series:
    """
    Load economic data from FRED (Federal Reserve Economic Data).
    
    Useful for macro features in HRM training.
    
    Args:
        series_id: FRED series ID (e.g., 'VIXCLS' for VIX, 'DGS10' for 10Y Treasury)
        start_date: Start date
        end_date: End date
    
    Returns:
        Economic data series
    """
    try:
        import fredapi
        fred = fredapi.Fred()
        
        data = fred.get_series(series_id, 
                              observation_start=start_date,
                              observation_end=end_date)
        return data.dropna()
        
    except ImportError:
        # Fallback to direct API call
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': 'your_fred_api_key',  # Free registration required
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        response = requests.get(base_url, params=params)
        data = response.json()
        
        observations = data['observations']
        values = []
        
        for obs in observations:
            if obs['value'] != '.':
                values.append({
                    'date': pd.to_datetime(obs['date']),
                    'value': float(obs['value'])
                })
        
        df = pd.DataFrame(values).set_index('date')
        return df['value']

def load_vix_data(start_date: str, end_date: str) -> pd.Series:
    """
    Load VIX data (CBOE Volatility Index) for free.
    
    VIX is crucial for options trading and regime detection.
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        VIX time series
    """
    try:
        # Try Yahoo Finance first (most reliable for VIX)
        vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
        return vix.dropna()
    
    except:
        # Fallback to FRED VIX data
        return load_fred_economic_data('VIXCLS', start_date, end_date)

# ================ ALTERNATIVE DATA SOURCES ================

def load_reddit_sentiment(subreddit: str = 'wallstreetbets', 
                         symbol: str = None,
                         days_back: int = 30) -> pd.DataFrame:
    """
    Load Reddit sentiment data as alternative data source.
    
    Note: This is a basic implementation. For production, consider
    paid sentiment analysis APIs like Sentieo or RavenPack.
    
    Args:
        subreddit: Reddit subreddit to analyze
        symbol: Stock symbol to filter for
        days_back: Number of days to look back
    
    Returns:
        DataFrame with sentiment scores by date
    """
    try:
        import praw
        from textblob import TextBlob
        
        # Reddit API setup (requires free reddit app registration)
        reddit = praw.Reddit(
            client_id='your_client_id',
            client_secret='your_client_secret',
            user_agent='hrm_trader_bot'
        )
        
        subreddit_obj = reddit.subreddit(subreddit)
        posts = []
        
        # Get recent posts
        for post in subreddit_obj.hot(limit=100):
            if symbol and symbol.upper() in post.title.upper():
                sentiment = TextBlob(post.title + ' ' + post.selftext).sentiment
                posts.append({
                    'date': pd.to_datetime(post.created_utc, unit='s').normalize(),
                    'sentiment_polarity': sentiment.polarity,
                    'sentiment_subjectivity': sentiment.subjectivity,
                    'score': post.score,
                    'num_comments': post.num_comments
                })
        
        if not posts:
            return pd.DataFrame()
        
        df = pd.DataFrame(posts)
        
        # Aggregate by date
        daily_sentiment = df.groupby('date').agg({
            'sentiment_polarity': 'mean',
            'sentiment_subjectivity': 'mean',
            'score': 'sum',
            'num_comments': 'sum'
        })
        
        return daily_sentiment
        
    except ImportError:
        logger.warning("Reddit sentiment requires 'praw' and 'textblob' packages")
        return pd.DataFrame()

def load_news_sentiment(symbol: str, api_key: str, days_back: int = 30) -> pd.DataFrame:
    """
    Load news sentiment using NewsAPI (free tier: 1000 requests/day).
    
    Args:
        symbol: Company symbol to search news for
        api_key: NewsAPI key (free from newsapi.org)
        days_back: Days to look back
    
    Returns:
        DataFrame with news sentiment scores
    """
    try:
        from datetime import datetime, timedelta
        from textblob import TextBlob
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        base_url = "https://newsapi.org/v2/everything"
        params = {
            'q': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'apiKey': api_key,
            'language': 'en'
        }
        
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data['status'] != 'ok':
            raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
        
        articles = data['articles']
        news_sentiment = []
        
        for article in articles:
            if article['title'] and article['description']:
                text = article['title'] + ' ' + article['description']
                sentiment = TextBlob(text).sentiment
                
                news_sentiment.append({
                    'date': pd.to_datetime(article['publishedAt']).normalize(),
                    'sentiment_polarity': sentiment.polarity,
                    'sentiment_subjectivity': sentiment.subjectivity,
                    'source': article['source']['name']
                })
        
        if not news_sentiment:
            return pd.DataFrame()
        
        df = pd.DataFrame(news_sentiment)
        
        # Aggregate by date
        daily_news = df.groupby('date').agg({
            'sentiment_polarity': 'mean',
            'sentiment_subjectivity': 'mean'
        })
        
        return daily_news
        
    except ImportError:
        logger.warning("News sentiment requires 'textblob' package")
        return pd.DataFrame()

# ================ CRYPTO OPTIONS DATA ================

def load_deribit_options(symbol: str = 'BTC', start_date: str = None, 
                        end_date: str = None, target_dte: int = 30) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load cryptocurrency options data from Deribit (free API).
    
    Useful for training HRM on crypto options markets which have
    different volatility characteristics than equity options.
    
    Args:
        symbol: Crypto symbol ('BTC' or 'ETH')
        start_date: Start date
        end_date: End date
        target_dte: Target DTE
    
    Returns:
        Tuple of (S, iv_entry, iv_exit, expiry) Series
    """
    base_url = "https://www.deribit.com/api/v2/public"
    
    try:
        # Get current instruments
        instruments_url = f"{base_url}/get_instruments"
        params = {
            'currency': symbol,
            'kind': 'option',
            'expired': False
        }
        
        response = requests.get(instruments_url, params=params)
        instruments = response.json()['result']
        
        # Filter for target DTE
        today = pd.Timestamp.now().normalize()
        target_instruments = []
        
        for inst in instruments:
            exp_date = pd.to_datetime(inst['expiration_timestamp'], unit='ms').normalize()
            dte = (exp_date - today).days
            
            if abs(dte - target_dte) <= 3:  # Within 3 days of target
                target_instruments.append(inst)
        
        if not target_instruments:
            raise ValueError(f"No {symbol} options found near {target_dte} DTE")
        
        # Get historical underlying price
        underlying_symbol = f"{symbol}-PERPETUAL"
        hist_url = f"{base_url}/get_tradingview_chart_data"
        hist_params = {
            'instrument_name': underlying_symbol,
            'start_timestamp': int(pd.to_datetime(start_date).timestamp() * 1000),
            'end_timestamp': int(pd.to_datetime(end_date).timestamp() * 1000),
            'resolution': '1D'
        }
        
        hist_response = requests.get(hist_url, params=hist_params)
        hist_data = hist_response.json()['result']
        
        # Parse price data
        prices = []
        for i, timestamp in enumerate(hist_data['ticks']):
            prices.append({
                'date': pd.to_datetime(timestamp, unit='ms').normalize(),
                'close': hist_data['close'][i]
            })
        
        price_df = pd.DataFrame(prices).set_index('date').sort_index()
        S = price_df['close']
        
        # Get ATM option IV (using current market data as proxy)
        atm_strike = round(S.iloc[-1] / 1000) * 1000  # Round to nearest 1000 for crypto
        
        iv_values = []
        for inst in target_instruments:
            if abs(inst['strike'] - atm_strike) < 500:  # Near ATM
                ticker_url = f"{base_url}/ticker"
                ticker_params = {'instrument_name': inst['instrument_name']}
                
                ticker_response = requests.get(ticker_url, params=ticker_params)
                ticker_data = ticker_response.json()['result']
                
                if ticker_data.get('mark_iv'):
                    iv_values.append(ticker_data['mark_iv'] / 100)  # Convert percentage
        
        if iv_values:
            avg_iv = np.mean(iv_values)
        else:
            # Fallback to historical volatility
            returns = np.log(S / S.shift(1)).dropna()
            avg_iv = returns.std() * np.sqrt(365)  # Crypto trades 365 days
        
        # Create IV series
        iv_entry = pd.Series(avg_iv, index=S.index, name='iv_entry')
        iv_exit = iv_entry.shift(-5).fillna(avg_iv)
        
        # Create expiry series
        exp_date = pd.to_datetime(target_instruments[0]['expiration_timestamp'], unit='ms').normalize()
        expiry = pd.Series(exp_date, index=S.index, name='expiry')
        
        return S, iv_entry, iv_exit, expiry
        
    except Exception as e:
        logger.error(f"Deribit options loader failed: {e}")
        raise

# ================ UTILITY FUNCTIONS ================

def combine_data_sources(sources: list[dict], symbol: str, start_date: str, 
                        end_date: str, target_dte: int = 30) -> dict[str, tuple]:
    """
    Combine multiple data sources for robust HRM training.
    
    Args:
        sources: List of data source configurations
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        target_dte: Target DTE
    
    Returns:
        Dictionary mapping source names to (S, iv_entry, iv_exit, expiry) tuples
    """
    combined_data = {}
    
    for source_config in sources:
        source_type = source_config['type']
        source_name = source_config.get('name', source_type)
        
        try:
            if source_type == 'yahoo':
                data = load_yahoo_finance_options(symbol, start_date, end_date, target_dte)
            elif source_type == 'alpha_vantage':
                data = load_alpha_vantage_data(symbol, source_config['api_key'], 
                                             start_date, end_date, target_dte)
            elif source_type == 'iex':
                data = load_iex_cloud_data(symbol, source_config['token'], 
                                         start_date, end_date, target_dte)
            elif source_type == 'deribit':
                data = load_deribit_options(symbol, start_date, end_date, target_dte)
            else:
                logger.warning(f"Unknown source type: {source_type}")
                continue
                
            combined_data[source_name] = data
            logger.info(f"Successfully loaded data from {source_name}")
            
        except Exception as e:
            logger.error(f"Failed to load data from {source_name}: {e}")
            continue
    
    return combined_data

def create_ensemble_features(combined_data: dict[str, tuple]) -> pd.DataFrame:
    """
    Create ensemble features from multiple data sources.
    
    Combines different IV estimates and creates consensus features
    for more robust HRM training.
    
    Args:
        combined_data: Dictionary of data from different sources
    
    Returns:
        DataFrame with ensemble features
    """
    all_features = []
    
    for source_name, (S, iv_entry, iv_exit, expiry) in combined_data.items():
        features = pd.DataFrame(index=S.index)
        features[f'{source_name}_price'] = S
        features[f'{source_name}_iv_entry'] = iv_entry
        features[f'{source_name}_iv_exit'] = iv_exit
        features[f'{source_name}_returns'] = np.log(S / S.shift(1))
        features[f'{source_name}_vol'] = features[f'{source_name}_returns'].rolling(20).std() * np.sqrt(252)
        
        all_features.append(features)
    
    if not all_features:
        return pd.DataFrame()
    
    # Combine all features
    ensemble_df = pd.concat(all_features, axis=1)
    
    # Create consensus features
    price_cols = [col for col in ensemble_df.columns if 'price' in col]
    iv_entry_cols = [col for col in ensemble_df.columns if 'iv_entry' in col]
    iv_exit_cols = [col for col in ensemble_df.columns if 'iv_exit' in col]
    
    if price_cols:
        ensemble_df['consensus_price'] = ensemble_df[price_cols].mean(axis=1)
    if iv_entry_cols:
        ensemble_df['consensus_iv_entry'] = ensemble_df[iv_entry_cols].mean(axis=1)
    if iv_exit_cols:
        ensemble_df['consensus_iv_exit'] = ensemble_df[iv_exit_cols].mean(axis=1)
    
    # Add spread measures (disagreement between sources)
    if len(price_cols) > 1:
        ensemble_df['price_spread'] = ensemble_df[price_cols].std(axis=1)
    if len(iv_entry_cols) > 1:
        ensemble_df['iv_entry_spread'] = ensemble_df[iv_entry_cols].std(axis=1)
    
    return ensemble_df

# ================ EXAMPLE CONFIGURATIONS ================

def get_free_data_config() -> list[dict]:
    """
    Get configuration for completely free data sources.
    
    Returns:
        List of free data source configurations
    """
    return [
        {'type': 'yahoo', 'name': 'yahoo_finance'},
        {'type': 'fred_vix', 'name': 'fed_vix'}
    ]

def get_freemium_data_config(alpha_vantage_key: str = None, 
                           iex_token: str = None,
                           news_api_key: str = None) -> list[dict]:
    """
    Get configuration for freemium data sources (free tiers).
    
    Args:
        alpha_vantage_key: Alpha Vantage API key
        iex_token: IEX Cloud token
        news_api_key: NewsAPI key
    
    Returns:
        List of freemium data source configurations
    """
    config = [
        {'type': 'yahoo', 'name': 'yahoo_finance'}
    ]
    
    if alpha_vantage_key:
        config.append({
            'type': 'alpha_vantage', 
            'name': 'alpha_vantage',
            'api_key': alpha_vantage_key
        })
    
    if iex_token:
        config.append({
            'type': 'iex',
            'name': 'iex_cloud', 
            'token': iex_token
        })
    
    return config

def get_crypto_data_config() -> list[dict]:
    """
    Get configuration for cryptocurrency options data.
    
    Returns:
        List of crypto data source configurations
    """
    return [
        {'type': 'deribit', 'name': 'deribit_btc'},
        {'type': 'yahoo', 'name': 'yahoo_crypto'}  # For underlying crypto prices
    ]