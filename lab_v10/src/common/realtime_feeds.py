"""
realtime_feeds.py
================

Real-time data feeds for HRM training and inference.
Provides streaming data capabilities from multiple sources
for live model training and backtesting.

Features:
- WebSocket data feeds
- REST API integrations
- Real-time feature engineering
- Data buffering and batching
- Multiple data source aggregation
- Fault tolerance and reconnection
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import aiohttp
import logging
from datetime import datetime, timedelta
import queue
import threading
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Standardized data point structure."""
    timestamp: datetime
    symbol: str
    data_type: str  # 'price', 'option', 'news', 'economic'
    value: Dict[str, Any]
    source: str

class DataFeed(ABC):
    """Abstract base class for data feeds."""
    
    def __init__(self, symbols: List[str], callback: Callable[[DataPoint], None] = None):
        self.symbols = symbols
        self.callback = callback
        self.is_running = False
        self.buffer = queue.Queue(maxsize=10000)
    
    @abstractmethod
    async def start(self):
        """Start the data feed."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the data feed."""
        pass
    
    def add_data_point(self, data_point: DataPoint):
        """Add a data point to the buffer and call callback if set."""
        try:
            self.buffer.put_nowait(data_point)
            if self.callback:
                self.callback(data_point)
        except queue.Full:
            logger.warning("Data buffer full, dropping oldest data")
            self.buffer.get_nowait()  # Remove oldest
            self.buffer.put_nowait(data_point)  # Add new
    
    def get_buffered_data(self, max_items: int = 100) -> List[DataPoint]:
        """Get buffered data points."""
        data_points = []
        for _ in range(min(max_items, self.buffer.qsize())):
            try:
                data_points.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        return data_points

class PolygonIOFeed(DataFeed):
    """Real-time data feed using Polygon.io WebSocket API."""
    
    def __init__(self, api_key: str, symbols: List[str], 
                 callback: Callable[[DataPoint], None] = None):
        super().__init__(symbols, callback)
        self.api_key = api_key
        self.websocket = None
        self.base_url = "wss://socket.polygon.io/stocks"
    
    async def start(self):
        """Start Polygon.io WebSocket connection."""
        self.is_running = True
        
        try:
            async with websockets.connect(self.base_url) as websocket:
                self.websocket = websocket
                
                # Authenticate
                auth_msg = {"action": "auth", "params": self.api_key}
                await websocket.send(json.dumps(auth_msg))
                
                # Subscribe to symbols
                for symbol in self.symbols:
                    subscribe_msg = {
                        "action": "subscribe",
                        "params": f"T.{symbol},Q.{symbol},A.{symbol}"  # Trades, Quotes, Aggregates
                    }
                    await websocket.send(json.dumps(subscribe_msg))
                
                logger.info(f"Connected to Polygon.io for symbols: {self.symbols}")
                
                # Listen for messages
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        await self._process_message(json.loads(message))
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing Polygon.io message: {e}")
                        
        except Exception as e:
            logger.error(f"Polygon.io connection error: {e}")
            self.is_running = False
    
    async def stop(self):
        """Stop the WebSocket connection."""
        self.is_running = False
        if self.websocket:
            await self.websocket.close()
    
    async def _process_message(self, message: List[Dict]):
        """Process incoming WebSocket message."""
        for msg in message:
            msg_type = msg.get('ev')  # Event type
            
            if msg_type == 'T':  # Trade
                data_point = DataPoint(
                    timestamp=datetime.fromtimestamp(msg['t'] / 1000),
                    symbol=msg['sym'],
                    data_type='trade',
                    value={
                        'price': msg['p'],
                        'size': msg['s'],
                        'conditions': msg.get('c', [])
                    },
                    source='polygon'
                )
                self.add_data_point(data_point)
            
            elif msg_type == 'Q':  # Quote
                data_point = DataPoint(
                    timestamp=datetime.fromtimestamp(msg['t'] / 1000),
                    symbol=msg['sym'],
                    data_type='quote',
                    value={
                        'bid': msg['bp'],
                        'ask': msg['ap'],
                        'bid_size': msg['bs'],
                        'ask_size': msg['as']
                    },
                    source='polygon'
                )
                self.add_data_point(data_point)

class AlphaVantageFeed(DataFeed):
    """Real-time data feed using Alpha Vantage API."""
    
    def __init__(self, api_key: str, symbols: List[str],
                 callback: Callable[[DataPoint], None] = None):
        super().__init__(symbols, callback)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
        self.update_interval = 60  # seconds
    
    async def start(self):
        """Start Alpha Vantage polling."""
        self.is_running = True
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            while self.is_running:
                for symbol in self.symbols:
                    try:
                        await self._fetch_symbol_data(symbol)
                    except Exception as e:
                        logger.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
    
    async def stop(self):
        """Stop the data feed."""
        self.is_running = False
    
    async def _fetch_symbol_data(self, symbol: str):
        """Fetch data for a specific symbol."""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        async with self.session.get(self.base_url, params=params) as response:
            data = await response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                
                data_point = DataPoint(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    data_type='quote',
                    value={
                        'price': float(quote['05. price']),
                        'change': float(quote['09. change']),
                        'change_percent': quote['10. change percent'].rstrip('%'),
                        'volume': int(quote['06. volume'])
                    },
                    source='alpha_vantage'
                )
                self.add_data_point(data_point)

class YahooFinanceFeed(DataFeed):
    """Real-time data feed using Yahoo Finance (free)."""
    
    def __init__(self, symbols: List[str], callback: Callable[[DataPoint], None] = None):
        super().__init__(symbols, callback)
        self.update_interval = 30  # seconds (be respectful to free API)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def start(self):
        """Start Yahoo Finance polling."""
        self.is_running = True
        
        while self.is_running:
            # Fetch data for all symbols in parallel
            loop = asyncio.get_event_loop()
            
            tasks = []
            for symbol in self.symbols:
                task = loop.run_in_executor(self.executor, self._fetch_symbol_data, symbol)
                tasks.append(task)
            
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error in Yahoo Finance batch fetch: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    async def stop(self):
        """Stop the data feed."""
        self.is_running = False
        self.executor.shutdown(wait=True)
    
    def _fetch_symbol_data(self, symbol: str):
        """Fetch data for a specific symbol (runs in thread)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            if current_price is None:
                # Fallback to recent history
                hist = ticker.history(period='1d', interval='1m')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
            
            if current_price is not None:
                data_point = DataPoint(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    data_type='quote',
                    value={
                        'price': float(current_price),
                        'volume': info.get('volume', 0),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                        '52_week_low': info.get('fiftyTwoWeekLow', 0)
                    },
                    source='yahoo_finance'
                )
                self.add_data_point(data_point)
                
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Yahoo Finance: {e}")

class NewsFeed(DataFeed):
    """Real-time news feed for sentiment analysis."""
    
    def __init__(self, api_key: str, symbols: List[str],
                 callback: Callable[[DataPoint], None] = None):
        super().__init__(symbols, callback)
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.session = None
        self.update_interval = 300  # 5 minutes
        self.last_fetch = {}
    
    async def start(self):
        """Start news feed polling."""
        self.is_running = True
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            while self.is_running:
                for symbol in self.symbols:
                    try:
                        await self._fetch_news(symbol)
                    except Exception as e:
                        logger.error(f"Error fetching news for {symbol}: {e}")
                
                await asyncio.sleep(self.update_interval)
    
    async def stop(self):
        """Stop the news feed."""
        self.is_running = False
    
    async def _fetch_news(self, symbol: str):
        """Fetch news for a specific symbol."""
        # Only fetch new articles since last update
        from_date = self.last_fetch.get(symbol, datetime.now() - timedelta(hours=1))
        
        params = {
            'q': f'"{symbol}"',
            'from': from_date.isoformat(),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.api_key
        }
        
        async with self.session.get(self.base_url, params=params) as response:
            data = await response.json()
            
            if data['status'] == 'ok':
                for article in data['articles']:
                    # Simple sentiment analysis
                    sentiment_score = self._analyze_sentiment(
                        article['title'] + ' ' + (article['description'] or '')
                    )
                    
                    data_point = DataPoint(
                        timestamp=datetime.fromisoformat(article['publishedAt'].rstrip('Z')),
                        symbol=symbol,
                        data_type='news',
                        value={
                            'title': article['title'],
                            'description': article['description'],
                            'source': article['source']['name'],
                            'url': article['url'],
                            'sentiment_score': sentiment_score
                        },
                        source='newsapi'
                    )
                    self.add_data_point(data_point)
                
                self.last_fetch[symbol] = datetime.now()
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (replace with proper NLP in production)."""
        if not text:
            return 0.0
        
        # Simple keyword-based sentiment
        positive_words = ['profit', 'gain', 'up', 'bullish', 'strong', 'good', 'positive', 'growth']
        negative_words = ['loss', 'down', 'bearish', 'weak', 'bad', 'negative', 'decline', 'fall']
        
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        total_words = len(text.split())
        sentiment = (pos_count - neg_count) / max(total_words, 1)
        
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]

class RealTimeFeatureEngine:
    """Real-time feature engineering engine."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.price_history = {}
        self.volume_history = {}
        self.news_sentiment = {}
        self.features = {}
    
    def process_data_point(self, data_point: DataPoint) -> Dict[str, float]:
        """
        Process a data point and return computed features.
        
        Args:
            data_point: Incoming data point
        
        Returns:
            Dictionary of computed features
        """
        symbol = data_point.symbol
        
        # Initialize history for new symbols
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.news_sentiment[symbol] = []
            self.features[symbol] = {}
        
        features = {}
        
        if data_point.data_type == 'quote' or data_point.data_type == 'trade':
            # Update price history
            price = data_point.value.get('price', 0)
            volume = data_point.value.get('volume', 0) or data_point.value.get('size', 0)
            
            self.price_history[symbol].append((data_point.timestamp, price))
            self.volume_history[symbol].append((data_point.timestamp, volume))
            
            # Trim history to window size
            self.price_history[symbol] = self.price_history[symbol][-self.window_size:]
            self.volume_history[symbol] = self.volume_history[symbol][-self.window_size:]
            
            # Compute price-based features
            if len(self.price_history[symbol]) >= 2:
                prices = [p[1] for p in self.price_history[symbol]]
                features.update(self._compute_price_features(prices))
            
            # Compute volume-based features
            if len(self.volume_history[symbol]) >= 2:
                volumes = [v[1] for v in self.volume_history[symbol]]
                features.update(self._compute_volume_features(volumes))
        
        elif data_point.data_type == 'news':
            # Update sentiment history
            sentiment = data_point.value.get('sentiment_score', 0)
            self.news_sentiment[symbol].append((data_point.timestamp, sentiment))
            self.news_sentiment[symbol] = self.news_sentiment[symbol][-100:]  # Keep last 100 news items
            
            # Compute sentiment features
            if self.news_sentiment[symbol]:
                sentiments = [s[1] for s in self.news_sentiment[symbol]]
                features.update(self._compute_sentiment_features(sentiments))
        
        # Update stored features for this symbol
        self.features[symbol].update(features)
        
        return features
    
    def _compute_price_features(self, prices: List[float]) -> Dict[str, float]:
        """Compute price-based technical features."""
        if len(prices) < 2:
            return {}
        
        prices_array = np.array(prices)
        
        features = {
            'price_current': prices[-1],
            'price_change': prices[-1] - prices[-2] if len(prices) >= 2 else 0,
            'price_change_pct': (prices[-1] / prices[-2] - 1) * 100 if len(prices) >= 2 and prices[-2] != 0 else 0
        }
        
        # Moving averages
        if len(prices) >= 5:
            features['sma_5'] = np.mean(prices[-5:])
            features['price_vs_sma5'] = (prices[-1] / features['sma_5'] - 1) * 100
        
        if len(prices) >= 20:
            features['sma_20'] = np.mean(prices[-20:])
            features['price_vs_sma20'] = (prices[-1] / features['sma_20'] - 1) * 100
            
            # Bollinger Bands
            sma20 = features['sma_20']
            std20 = np.std(prices[-20:])
            features['bb_upper'] = sma20 + 2 * std20
            features['bb_lower'] = sma20 - 2 * std20
            features['bb_position'] = (prices[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volatility
        if len(prices) >= 10:
            returns = np.diff(np.log(prices[-10:]))
            features['volatility_10'] = np.std(returns) * np.sqrt(252) * 100  # Annualized %
        
        # RSI
        if len(prices) >= 14:
            gains = []
            losses = []
            for i in range(1, min(15, len(prices))):
                change = prices[-i] - prices[-i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
        
        return features
    
    def _compute_volume_features(self, volumes: List[float]) -> Dict[str, float]:
        """Compute volume-based features."""
        if len(volumes) < 2:
            return {}
        
        volumes_array = np.array(volumes)
        
        features = {
            'volume_current': volumes[-1],
            'volume_change_pct': (volumes[-1] / volumes[-2] - 1) * 100 if volumes[-2] != 0 else 0
        }
        
        # Volume moving averages
        if len(volumes) >= 5:
            features['volume_sma_5'] = np.mean(volumes[-5:])
            features['volume_vs_sma5'] = (volumes[-1] / features['volume_sma_5'] - 1) * 100 if features['volume_sma_5'] != 0 else 0
        
        if len(volumes) >= 20:
            features['volume_sma_20'] = np.mean(volumes[-20:])
            features['volume_vs_sma20'] = (volumes[-1] / features['volume_sma_20'] - 1) * 100 if features['volume_sma_20'] != 0 else 0
        
        return features
    
    def _compute_sentiment_features(self, sentiments: List[float]) -> Dict[str, float]:
        """Compute sentiment-based features."""
        if not sentiments:
            return {}
        
        features = {
            'sentiment_current': sentiments[-1],
            'sentiment_avg_1h': np.mean(sentiments[-12:]),  # Assuming 5-min intervals
            'sentiment_avg_24h': np.mean(sentiments)
        }
        
        return features
    
    def get_feature_vector(self, symbol: str) -> np.ndarray:
        """Get feature vector for a symbol."""
        if symbol not in self.features:
            return np.array([])
        
        feature_dict = self.features[symbol]
        
        # Define feature order for consistent vector
        feature_names = [
            'price_current', 'price_change', 'price_change_pct',
            'sma_5', 'price_vs_sma5', 'sma_20', 'price_vs_sma20',
            'bb_position', 'volatility_10', 'rsi',
            'volume_current', 'volume_change_pct', 'volume_vs_sma5', 'volume_vs_sma20',
            'sentiment_current', 'sentiment_avg_1h', 'sentiment_avg_24h'
        ]
        
        vector = []
        for name in feature_names:
            vector.append(feature_dict.get(name, 0.0))
        
        return np.array(vector)

class RealTimeDataManager:
    """Manage multiple real-time data feeds."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.feeds = {}
        self.feature_engine = RealTimeFeatureEngine()
        self.callbacks = []
        self.is_running = False
    
    def add_feed(self, name: str, feed: DataFeed):
        """Add a data feed."""
        feed.callback = self._on_data_point
        self.feeds[name] = feed
    
    def add_callback(self, callback: Callable[[str, Dict[str, float]], None]):
        """Add a callback for processed features."""
        self.callbacks.append(callback)
    
    def _on_data_point(self, data_point: DataPoint):
        """Handle incoming data point."""
        # Process through feature engine
        features = self.feature_engine.process_data_point(data_point)
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(data_point.symbol, features)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def start_all_feeds(self):
        """Start all registered feeds."""
        self.is_running = True
        
        # Start all feeds concurrently
        tasks = []
        for name, feed in self.feeds.items():
            logger.info(f"Starting feed: {name}")
            task = asyncio.create_task(feed.start())
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in feed management: {e}")
        finally:
            self.is_running = False
    
    async def stop_all_feeds(self):
        """Stop all feeds."""
        self.is_running = False
        
        for name, feed in self.feeds.items():
            logger.info(f"Stopping feed: {name}")
            await feed.stop()
    
    def get_latest_features(self, symbol: str) -> Dict[str, float]:
        """Get latest features for a symbol."""
        return self.feature_engine.features.get(symbol, {})
    
    def get_feature_vector(self, symbol: str) -> np.ndarray:
        """Get feature vector for a symbol."""
        return self.feature_engine.get_feature_vector(symbol)

# Example configuration and usage
def create_realtime_config() -> Dict:
    """Create example real-time data configuration."""
    return {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'SPY'],
        'feeds': {
            'yahoo_finance': {
                'enabled': True,
                'type': 'yahoo'
            },
            'alpha_vantage': {
                'enabled': False,  # Requires API key
                'type': 'alpha_vantage',
                'api_key': 'your_api_key_here'
            },
            'news': {
                'enabled': False,  # Requires API key
                'type': 'news',
                'api_key': 'your_newsapi_key_here'
            }
        },
        'feature_window': 1000,
        'update_interval': 30
    }

async def main_example():
    """Example of using the real-time data system."""
    config = create_realtime_config()
    symbols = config['symbols']
    
    # Create data manager
    data_manager = RealTimeDataManager(symbols)
    
    # Add callback to print features
    def on_features(symbol: str, features: Dict[str, float]):
        if features:
            print(f"{symbol}: Price={features.get('price_current', 0):.2f}, "
                  f"Change={features.get('price_change_pct', 0):.2f}%")
    
    data_manager.add_callback(on_features)
    
    # Add Yahoo Finance feed (free)
    yahoo_feed = YahooFinanceFeed(symbols)
    data_manager.add_feed('yahoo', yahoo_feed)
    
    # Start feeds
    try:
        logger.info("Starting real-time data feeds...")
        await data_manager.start_all_feeds()
    except KeyboardInterrupt:
        logger.info("Stopping feeds...")
        await data_manager.stop_all_feeds()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_example())