import zmq
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import threading
import queue
import lightgbm as lgb
from loguru import logger
import gc
import torch
import os
import time
import mmap
import struct
from ctypes import *
import uuid
import sys

# Add the SignalData and ShmHeader structures from signal_generator.py
class SignalData(Structure):
    _fields_ = [
        ("symbol", c_char * 16),
        ("side", c_char * 8),
        ("order_type", c_char * 16),
        ("quantity", c_double),
        ("price", c_double),
        ("timestamp", c_double),
        ("action", c_char * 8),
        ("orderId", c_long),
        ("clientOrderId", c_char * 32),
        ("with_tp_sl", c_bool)
    ]

class ShmHeader(Structure):
    _fields_ = [
        ("n", c_int),  # atomic counter
        ("start_ts", c_double),
        ("interval", c_double),
        ("limit", c_int)
    ]

# Add order sending function from signal_generator.py
def send_order_signal(symbol, action, side="", order_type="", quantity=0.0, price=0.0, 
                     order_id=0, client_order_id="", with_tp_sl=False):
    try:
        # Open shared memory
        shm_fd = os.open('/dev/shm/binance_signals', os.O_RDWR)
        
        # Get file size
        file_size = os.fstat(shm_fd).st_size
        shm = mmap.mmap(shm_fd, file_size, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        
        # Read header
        header = ShmHeader.from_buffer_copy(shm.read(sizeof(ShmHeader)))
        
        # Create signal data
        signal = SignalData(
            symbol=symbol.encode('utf-8'),
            side=side.encode('utf-8'),
            order_type=order_type.encode('utf-8'),
            quantity=quantity,
            price=price,
            timestamp=time.time(),
            action=action.encode('utf-8'),
            orderId=order_id,
            clientOrderId=client_order_id.encode('utf-8'),
            with_tp_sl=with_tp_sl
        )
        
        # Calculate position for new signal
        pos = sizeof(ShmHeader) + header.n * sizeof(SignalData)
        
        # Write signal to shared memory
        shm.seek(pos)
        shm.write(bytes(signal))
        
        # Update counter in header
        header.n += 1
        shm.seek(0)
        shm.write(bytes(header))
        
        # Cleanup
        shm.close()
        os.close(shm_fd)
        logger.info(f"{action} signal written successfully. Total signals: {header.n}")
        
    except Exception as e:
        logger.error(f"Error sending signal: {e}")

class OrderBookPredictor:
    def __init__(self, model_path: str, zmq_address: str = "tcp://localhost:5555", 
                 data_save_interval: int = 3600, confidence_threshold: float = 0.6,
                 trading_symbol: str = "DOGEUSDT", order_quantity_usdt: float = 10.0):
        """Initialize ZMQ subscriber, data structures and load the model"""
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(zmq_address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Data structures - updated window size to match training
        self.window_size = 1800  # 1800 data points (15 minutes at 500ms interval)
        self.data_window = deque(maxlen=self.window_size + 120)  # Add extra buffer for future predictions
        self.message_queue = queue.Queue()
        
        # Data storage parameters
        self.data_save_interval = data_save_interval  # How often to save data (in seconds)
        self.last_save_time = datetime.now()
        self.data_buffer = []  # Buffer to store data before saving
        self.data_dir = "collected_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Trading signal parameters
        self.confidence_threshold = confidence_threshold
        self.current_position = None  # None=no position, 'long'=long position, 'short'=short position
        self.trade_log = []  # Store trading decisions
        
        # Trading parameters
        self.trading_symbol = trading_symbol
        self.order_quantity_usdt = order_quantity_usdt
        self.active_orders = {}  # Track active orders by client_order_id
        
        # Load model
        logger.info("Loading model...")
        self.model = lgb.Booster(model_file=model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Extract model metadata
        self.feature_names = self.model.feature_name()
        logger.info(f"Model has {len(self.feature_names)} features")
        
        # Determine if model is classification or regression
        model_info = self.model.dump_model()
        self.objective = model_info.get('objective', '')
        self.is_classification = 'multiclass' in self.objective
        if self.is_classification:
            logger.info(f"Detected classification model with objective: {self.objective}")
            # Extract number of classes if available
            self.num_classes = model_info.get('num_class', 3)
            logger.info(f"Model has {self.num_classes} classes")
        else:
            logger.info(f"Detected regression model with objective: {self.objective}")
        
        # Try to load TCN model if available
        self.tcn_model = None
        tcn_model_path = 'tcn_model.pth'
        if os.path.exists(tcn_model_path):
            try:
                # Initialize dynamically based on available modules
                if 'OnlineTCN' in globals():
                    self.tcn_model = OnlineTCN(input_size=len(self.feature_names))
                    self.tcn_model.load_state_dict(torch.load(tcn_model_path))
                    self.tcn_model.eval()
                    logger.info("TCN model loaded successfully")
                else:
                    logger.warning("OnlineTCN class not found, skipping TCN model")
            except Exception as e:
                logger.error(f"Failed to load TCN model: {e}")
        
        # Control flags
        self.running = True
        self.process_thread = threading.Thread(target=self._process_messages)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Add counters for monitoring
        self.total_messages = 0
        self.valid_updates = 0
        self.last_log_time = datetime.now()
        
        # Add recent messages buffer
        self.recent_messages = deque(maxlen=5)
        
        # Define base feature columns that align with lgbm2.py
        self.base_feature_cols = [
            'relative_spread', 'depth_imbalance', 'bid_ask_slope',
            'order_book_pressure', 'weighted_price_depth', 'liquidity_imbalance',
            'flow_toxicity', 'price_momentum', 'volatility_ratio', 'ofi', 
            'vpin', 'pressure_change_rate', 'orderbook_gradient', 'depth_pressure_ratio'
        ]
        logger.info(f"Base feature columns: {self.base_feature_cols}")

    def calculate_features(self, data):
        """Calculate features from order book data following hft_pipeline.py logic"""
        bids = [(b['price'], b['quantity']) for b in data['bids']]
        asks = [(a['price'], a['quantity']) for a in data['asks']]
        
        # Ensure we have enough levels
        while len(bids) < 5:
            # Add fake level with same price but zero quantity
            last_price = bids[-1][0] if bids else 0
            bids.append((last_price - 0.01, 0))
            
        while len(asks) < 5:
            last_price = asks[-1][0] if asks else float('inf')
            asks.append((last_price + 0.01, 0))
        
        # Basic order book features
        mid_price = (asks[0][0] + bids[0][0]) / 2
        relative_spread = (asks[0][0] - bids[0][0]) / asks[0][0]
        
        # Avoid division by zero with epsilon
        epsilon = 1e-10
        
        # Depth imbalance with weighted approach (matching hft_pipeline.py)
        bid_depth = (bids[0][1] * 1.0 + 
                    bids[1][1] * 0.8 + 
                    bids[2][1] * 0.6 + 
                    bids[3][1] * 0.4 + 
                    bids[4][1] * 0.2)
        ask_depth = (asks[0][1] * 1.0 + 
                    asks[1][1] * 0.8 + 
                    asks[2][1] * 0.6 + 
                    asks[3][1] * 0.4 + 
                    asks[4][1] * 0.2)
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + epsilon)
        
        # Bid-ask slope calculation (following hft_pipeline.py)
        # Function for slope calculation similar to hft_pipeline.py
        def calculate_slope(prices, sizes, prev_prices=None, prev_sizes=None):
            if len(prices) < 2:
                return 0.0
            
            # Calculate spatial dimension price changes
            spatial_diffs = np.diff([p for p, _ in prices])
            spatial_weights = [s for _, s in prices[1:]]
            weight_sum = sum(spatial_weights) + epsilon
            spatial_weights = [w / weight_sum for w in spatial_weights]
            spatial_slope = sum(d * w for d, w in zip(spatial_diffs, spatial_weights))
            
            # We don't have previous data in this simple implementation
            return spatial_slope
        
        bid_slope = calculate_slope(bids, [q for _, q in bids])
        ask_slope = calculate_slope(asks, [q for _, q in asks])
        bid_ask_slope = (abs(bid_slope) + abs(ask_slope)) * np.sign(ask_slope - bid_slope)
        
        # Order book pressure
        bid_pressure = (bids[0][1] * bids[0][0] + bids[1][1] * bids[1][0])
        ask_pressure = (asks[0][1] * asks[0][0] + asks[1][1] * asks[1][0])
        order_book_pressure = np.log((bid_pressure + epsilon) / (ask_pressure + epsilon))
        
        # Weighted price depth
        weighted_bid = sum(p * v for p, v in bids[:3]) / (sum(v for _, v in bids[:3]) + epsilon)
        weighted_ask = sum(p * v for p, v in asks[:3]) / (sum(v for _, v in asks[:3]) + epsilon)
        weighted_price_depth = np.log((weighted_bid + epsilon) / (weighted_ask + epsilon))
        
        # Liquidity imbalance
        avg_bid_size = sum(v for _, v in bids[:3]) / 3
        avg_ask_size = sum(v for _, v in asks[:3]) / 3
        liquidity_imbalance = (avg_bid_size - avg_ask_size) / (avg_bid_size + avg_ask_size + epsilon)
        
        # Calculate advanced features if we have previous data
        if len(self.data_window) > 0:
            # Price momentum
            last_mid_price = self.data_window[-1]['mid_price']
            price_momentum = (mid_price - last_mid_price) / (last_mid_price + epsilon)
            
            # Volatility ratio
            realized_vol = 0.0
            if len(self.data_window) >= 20:
                # Calculate standard deviation of recent returns
                recent_prices = [entry['mid_price'] for entry in list(self.data_window)[-20:]]
                recent_returns = [(recent_prices[i+1]/recent_prices[i] - 1) for i in range(len(recent_prices)-1)]
                realized_vol = np.std(recent_returns) if recent_returns else 0
            
            # Use rolling mean of relative spread as implied volatility proxy
            implied_vol = np.mean([entry.get('relative_spread', 0) for entry in list(self.data_window)[-20:]]) + epsilon
            volatility_ratio = realized_vol / implied_vol
            
            # Flow toxicity
            flow_toxicity = depth_imbalance * relative_spread
            
            # Calculate OFI (Order Flow Imbalance)
            if 'bid_pressure' in self.data_window[-1] and 'ask_pressure' in self.data_window[-1]:
                last_bid_pressure = self.data_window[-1]['bid_pressure']
                last_ask_pressure = self.data_window[-1]['ask_pressure']
                bid_flow = max(0, bid_pressure - last_bid_pressure)
                ask_flow = max(0, last_ask_pressure - ask_pressure)
                ofi = (bid_flow - ask_flow) / (bid_flow + ask_flow + epsilon)
            else:
                ofi = 0.0
            
            # VPIN calculation
            vpin = 0.0
            if len(self.data_window) >= 50:
                ofi_values = [entry.get('ofi', 0) for entry in list(self.data_window)[-50:]]
                volumes = [entry.get('volume', 1) for entry in list(self.data_window)[-50:]]
                vpin = np.mean(ofi_values) * np.mean(volumes)
            
            # Pressure change rate
            if 'bid_pressure' in self.data_window[-1] and 'ask_pressure' in self.data_window[-1]:
                last_bid_pressure = self.data_window[-1]['bid_pressure'] 
                last_ask_pressure = self.data_window[-1]['ask_pressure']
                bid_pressure_change = (bid_pressure - last_bid_pressure) / (last_bid_pressure + epsilon)
                ask_pressure_change = (ask_pressure - last_ask_pressure) / (last_ask_pressure + epsilon)
                pressure_change_rate = bid_pressure_change - ask_pressure_change
            else:
                pressure_change_rate = 0.0
                
            # Orderbook gradient (simplified from hft_pipeline.py)
            orderbook_gradient = 0.0
            if len(self.data_window) > 1:
                # Simple gradient based on price changes
                bid_prices = [b[0] for b in bids]
                ask_prices = [a[0] for a in asks]
                bid_gradient = np.mean(np.gradient(bid_prices)) if len(bid_prices) > 1 else 0
                ask_gradient = np.mean(np.gradient(ask_prices)) if len(ask_prices) > 1 else 0
                orderbook_gradient = (ask_gradient - bid_gradient) * (1 + abs(ask_gradient + bid_gradient))
            
            # Depth pressure ratio
            bid_pressure_sum = sum(p * s for p, s in bids)
            ask_pressure_sum = sum(p * s for p, s in asks)
            depth_pressure_ratio = np.log((bid_pressure_sum + epsilon) / (ask_pressure_sum + epsilon))
            
            # Add new features from lgbm2.py
            # Price momentum rolling window features
            price_momentum_1min = np.mean([entry.get('price_momentum', 0) for entry in list(self.data_window)[-120:]]) if len(self.data_window) >= 120 else price_momentum
            price_momentum_5min = np.mean([entry.get('price_momentum', 0) for entry in list(self.data_window)[-600:]]) if len(self.data_window) >= 600 else price_momentum
            
            # Volatility features
            volatility_1min = np.std([entry.get('mid_price', mid_price) for entry in list(self.data_window)[-120:]]) if len(self.data_window) >= 120 else realized_vol
            volatility_5min = np.std([entry.get('mid_price', mid_price) for entry in list(self.data_window)[-600:]]) if len(self.data_window) >= 600 else realized_vol
            
            # Volume pressure
            volume_pressure = 0.0
            if len(self.data_window) >= 120:
                ofi_sum = sum([entry.get('ofi', 0) for entry in list(self.data_window)[-120:]])
                vpin_mean = np.mean([entry.get('vpin', epsilon) for entry in list(self.data_window)[-120:]])
                volume_pressure = ofi_sum / vpin_mean
            
            # Trend strength
            trend_strength = price_momentum_5min / (volatility_5min + epsilon)
            
            # Spread and depth volatility
            spread_volatility = np.std([entry.get('relative_spread', 0) for entry in list(self.data_window)[-120:]]) if len(self.data_window) >= 120 else 0
            depth_volatility = np.std([entry.get('depth_imbalance', 0) for entry in list(self.data_window)[-120:]]) if len(self.data_window) >= 120 else 0
            
            # Pressure momentum
            pressure_momentum = order_book_pressure * price_momentum
        else:
            # Default values when we don't have enough history
            price_momentum = 0.0
            volatility_ratio = 0.0
            flow_toxicity = 0.0
            ofi = 0.0
            vpin = 0.0
            pressure_change_rate = 0.0
            orderbook_gradient = 0.0
            depth_pressure_ratio = 0.0
            price_momentum_1min = 0.0
            price_momentum_5min = 0.0
            volatility_1min = 0.0
            volatility_5min = 0.0
            volume_pressure = 0.0
            trend_strength = 0.0
            spread_volatility = 0.0
            depth_volatility = 0.0
            pressure_momentum = 0.0
        
        # Store additional data for next calculations
        volume = sum(q for _, q in bids[:3]) + sum(q for _, q in asks[:3])
        
        # Return complete feature dictionary
        return {
            'mid_price': mid_price,
            'relative_spread': relative_spread,
            'depth_imbalance': depth_imbalance,
            'bid_ask_slope': bid_ask_slope,
            'order_book_pressure': order_book_pressure,
            'weighted_price_depth': weighted_price_depth,
            'liquidity_imbalance': liquidity_imbalance,
            'flow_toxicity': flow_toxicity,
            'price_momentum': price_momentum,
            'volatility_ratio': volatility_ratio,
            'ofi': ofi,
            'vpin': vpin,
            'pressure_change_rate': pressure_change_rate,
            'orderbook_gradient': orderbook_gradient,
            'depth_pressure_ratio': depth_pressure_ratio,
            'bid_pressure': bid_pressure,
            'ask_pressure': ask_pressure,
            'volume': volume,
            # Add new features from lgbm2.py
            'price_momentum_1min': price_momentum_1min,
            'price_momentum_5min': price_momentum_5min,
            'volatility_1min': volatility_1min,
            'volatility_5min': volatility_5min, 
            'volume_pressure': volume_pressure,
            'trend_strength': trend_strength,
            'spread_volatility': spread_volatility,
            'depth_volatility': depth_volatility,
            'pressure_momentum': pressure_momentum
        }

    def create_features(self, data_window):
        """Create features following lgbm2.py logic for prediction"""
        if len(data_window) < self.window_size:
            logger.warning(f"Not enough data for prediction: {len(data_window)}/{self.window_size}")
            return None, None
        
        # Create DataFrame from the data window
        df = pd.DataFrame(data_window)
        
        # Basic feature columns based on lgbm2.py
        feature_cols = self.base_feature_cols
        
        # Ensure all required features exist
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.error(f"Missing required feature columns: {missing_features}")
            logger.info(f"Available columns: {df.columns.tolist()}")
            return None, None
        
        # Convert features to float32 for consistent processing
        numeric_df = df[feature_cols].astype(np.float32)
        
        # Data validation
        if numeric_df.isna().any().any():
            logger.warning("Found NaN values in features, performing forward fill")
            numeric_df = numeric_df.fillna(method='ffill').fillna(0)
        
        # Handle infinity values
        numeric_df = numeric_df.replace([np.inf, -np.inf], 0)
        
        # Extract features for prediction - align with lgbm2.py's feature structure
        features = numeric_df.values[-self.window_size:]
        
        # Validate shape before flattening
        if features.shape[0] != self.window_size:
            logger.error(f"Feature window size mismatch! Expected {self.window_size}, got {features.shape[0]}")
            return None, None
            
        # Flatten the features
        X = features.flatten()
        
        # Validate feature dimensions against what the model expects
        expected_features = len(self.feature_names)
        actual_features = X.shape[0]
        
        if actual_features != expected_features:
            logger.error(f"Feature dimension mismatch! Model expects {expected_features}, actual {actual_features}")
            logger.error(f"This usually indicates the feature creation process doesn't match training")
            
            # Additional debugging information
            logger.error(f"Model feature names: {self.feature_names}")
            logger.error(f"Current feature shape: {features.shape}")
            
            return None, None
        
        # Ensure data type is consistent with training
        X = X.astype(np.float32)
        
        return X.reshape(1, -1), None

    def save_data_to_file(self):
        """Save collected data to a file"""
        if not self.data_buffer:
            logger.info("No data to save")
            return
        
        try:
            # Create DataFrame from buffer
            df = pd.DataFrame(self.data_buffer)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/orderbook_data_{timestamp}.feather"
            
            # Save to feather format (compatible with your training code)
            df.reset_index().to_feather(filename)
            
            logger.info(f"Saved {len(df)} data points to {filename}")
            
            # Clear buffer after saving
            self.data_buffer = []
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def generate_trading_signal(self, prediction, timestamp, current_price):
        """Generate explicit trading signals based on model predictions"""
        signal = {
            'timestamp': timestamp,
            'price': current_price,
            'action': 'HOLD',  # Default action
            'confidence': 0.0,
            'prediction': None
        }
        
        try:
            if self.is_classification:
                # Classification model processing (3 classes: 0=down, 1=neutral, 2=up)
                class_probs = prediction[0] if len(prediction.shape) > 1 else prediction
                pred_class = np.argmax(class_probs)
                confidence = class_probs[pred_class]
                
                # Store prediction details
                signal['prediction'] = {
                    'class': int(pred_class),
                    'probabilities': {
                        'down': float(class_probs[0]),
                        'neutral': float(class_probs[1]),
                        'up': float(class_probs[2])
                    }
                }
                signal['confidence'] = float(confidence)
                
                # Generate trading signal based on prediction and confidence
                if confidence >= self.confidence_threshold:
                    if pred_class == 0:  # Down prediction
                        signal['action'] = 'SHORT'
                    elif pred_class == 2:  # Up prediction
                        signal['action'] = 'LONG'
                    # Class 1 (neutral) maintains HOLD
            else:
                # Regression model processing (continuous return prediction)
                return_pred = prediction[0]
                
                # Store prediction details
                signal['prediction'] = {
                    'predicted_return': float(return_pred)
                }
                
                # Use absolute return as confidence
                confidence = abs(return_pred)
                signal['confidence'] = float(confidence)
                
                # Generate trading signal based on prediction and threshold
                if confidence >= self.confidence_threshold:
                    if return_pred > 0:
                        signal['action'] = 'LONG'
                    elif return_pred < 0:
                        signal['action'] = 'SHORT'
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return signal

    def _process_messages(self):
        """Process incoming messages, store data, and generate trading signals"""
        last_prediction_time = None
        
        while self.running:
            try:
                message = self.socket.recv_string()
                self.total_messages += 1
                data = json.loads(message)
                
                # Record reception time for all messages
                current_time = datetime.now()
                
                # Process order book data
                if data.get('type') == 'depth' and data.get('bids') and data.get('asks'):
                    self.valid_updates += 1
                    timestamp = datetime.fromtimestamp(data['timestamp'] / 1e9)
                    
                    # Store recent messages (with best bid/ask prices)
                    self.recent_messages.append({
                        'timestamp': timestamp,
                        'best_bid': data['bids'][0]['price'],
                        'best_ask': data['asks'][0]['price'],
                        'mid_price': (data['bids'][0]['price'] + data['asks'][0]['price']) / 2
                    })
                    
                    # Calculate features and add to window
                    features = self.calculate_features(data)
                    features['Timestamp'] = timestamp
                    
                    # Add data to both window and buffer
                    self.data_window.append(features)
                    self.data_buffer.append(features)
                    
                    # Make prediction when enough data and time has passed
                    if (len(self.data_window) >= self.window_size and
                        (last_prediction_time is None or 
                         timestamp - last_prediction_time >= timedelta(seconds=10))):
                        
                        # Get current price for trading
                        current_price = features['mid_price']
                        
                        # Generate trading signal
                        signal = self.generate_trading_signal_from_data(timestamp, current_price)
                        
                        # Log the signal
                        if signal and signal['action'] != 'HOLD':
                            logger.info(f"\n=== TRADING SIGNAL ===")
                            logger.info(f"Time: {signal['timestamp']}")
                            logger.info(f"Action: {signal['action']}")
                            logger.info(f"Price: {signal['price']:.8f}")
                            logger.info(f"Confidence: {signal['confidence']:.4f}")
                            logger.info(f"====================\n")
                            
                            # Add to trade log
                            self.trade_log.append(signal)
                        
                        last_prediction_time = timestamp
                    
                    # Save data periodically
                    if (current_time - self.last_save_time).total_seconds() >= self.data_save_interval:
                        self.save_data_to_file()
                        self.last_save_time = current_time
                
                # Log system status every 30 seconds
                if (current_time - self.last_log_time).total_seconds() >= 30:
                    logger.info("\n=== System Status ===")
                    logger.info(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"Total Messages: {self.total_messages}")
                    logger.info(f"Valid Updates: {self.valid_updates}")
                    logger.info(f"Data Window Size: {len(self.data_window)}/{self.window_size + 120}")
                    logger.info(f"Data Buffer Size: {len(self.data_buffer)}")
                    logger.info(f"Trade Signals Generated: {len(self.trade_log)}")
                    
                    if self.recent_messages:
                        logger.info("\nLast 5 Order Book Updates:")
                        for msg in reversed(self.recent_messages):
                            logger.info(
                                f"[{msg['timestamp'].strftime('%H:%M:%S.%f')[:-3]}] "
                                f"Bid: {msg['best_bid']:.2f} | "
                                f"Ask: {msg['best_ask']:.2f} | "
                                f"Mid: {msg['mid_price']:.2f}"
                            )
                    else:
                        logger.info("\nNo order book updates received yet")
                    
                    self.last_log_time = current_time
                    logger.info("=== End of Status ===\n")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                logger.exception("Error details:")

    def place_order(self, action, price):
        """Place a limit order based on prediction signal"""
        try:
            # Determine order side based on action
            if action == 'LONG':
                side = 'BUY'
            elif action == 'SHORT':
                side = 'SELL'
            else:
                logger.warning(f"Ignoring non-tradable action: {action}")
                return None
            
            # Calculate quantity based on fixed USDT amount
            # For simplicity, calculate quantity as USDT amount / price
            quantity = self.order_quantity_usdt / price
            
            # Generate a unique client order ID
            client_order_id = f"pred_{uuid.uuid4().hex[:16]}"
            
            # Place the order
            logger.info(f"Placing a {side} limit order for {quantity:.4f} {self.trading_symbol} at {price:.4f} with TP/SL...")
            send_order_signal(
                symbol=self.trading_symbol,
                side=side,
                order_type="LIMIT",
                quantity=quantity,
                price=price,
                action="NEW",
                client_order_id=client_order_id,
                with_tp_sl=True  # Enable TP/SL for automatic trading
            )
            
            # Track this order
            self.active_orders[client_order_id] = {
                'symbol': self.trading_symbol,
                'side': side,
                'price': price,
                'quantity': quantity,
                'timestamp': datetime.now(),
                'with_tp_sl': True
            }
            
            return client_order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            logger.exception("Order placement error details:")
            return None
            
    def cancel_all_orders(self):
        """Cancel all active orders"""
        for client_order_id in list(self.active_orders.keys()):
            try:
                logger.info(f"Cancelling order: {client_order_id}")
                send_order_signal(
                    symbol=self.trading_symbol,
                    action="CANCEL",
                    client_order_id=client_order_id
                )
                # Remove from tracking after cancellation request
                self.active_orders.pop(client_order_id, None)
            except Exception as e:
                logger.error(f"Error cancelling order {client_order_id}: {e}")

    def generate_trading_signal_from_data(self, timestamp, current_price):
        """Generate a trading signal based on current data"""
        try:
            # Create features
            X, _ = self.create_features(list(self.data_window))
            
            if X is None:
                logger.warning("Not enough data for prediction")
                return None
            
            # Make prediction
            raw_pred = self.model.predict(X)
            
            # Generate trading signal
            signal = self.generate_trading_signal(raw_pred, timestamp, current_price)
            
            # Execute trading action if signal generated
            if signal and signal['action'] != 'HOLD':
                # Place order at mid price
                client_order_id = self.place_order(signal['action'], current_price)
                
                if client_order_id:
                    # Add order ID to the signal for reference
                    signal['client_order_id'] = client_order_id
                    logger.info(f"Order placed with client ID: {client_order_id}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            logger.exception("Prediction error details:")
            return None
    
    def stop(self):
        """Stop the predictor and clean up resources"""
        self.running = False
        
        # Cancel any active orders before shutting down
        logger.info("Cancelling all active orders before shutdown...")
        self.cancel_all_orders()
        
        # Save any remaining data
        self.save_data_to_file()
        
        # Save trade log
        if self.trade_log:
            try:
                trade_df = pd.DataFrame(self.trade_log)
                trade_df.to_csv(f"{self.data_dir}/trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
                logger.info(f"Saved {len(trade_df)} trade signals to log file")
            except Exception as e:
                logger.error(f"Error saving trade log: {e}")
        
        # Wait for thread to complete
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
            
        # Clean up ZMQ resources
        self.socket.close()
        self.context.term()
        logger.info("Predictor stopped and resources cleaned up")

def main():
    # Use argparse to allow command-line configuration
    import argparse
    parser = argparse.ArgumentParser(description='Order Book Predictor')
    parser.add_argument('--model', type=str, default='incremental_model.bin', help='Path to model file')
    parser.add_argument('--zmq_address', type=str, default='tcp://localhost:5555', help='ZMQ socket address')
    parser.add_argument('--save_interval', type=int, default=3600, help='Data save interval in seconds')
    parser.add_argument('--threshold', type=float, default=0.6, help='Confidence threshold for trading signals')
    parser.add_argument('--symbol', type=str, default='DOGEUSDT', help='Trading symbol')
    parser.add_argument('--order_size', type=float, default=10.0, help='Order size in USDT')
    args = parser.parse_args()
    
    predictor = OrderBookPredictor(
        model_path=args.model,
        zmq_address=args.zmq_address,
        data_save_interval=args.save_interval,
        confidence_threshold=args.threshold,
        trading_symbol=args.symbol,
        order_quantity_usdt=args.order_size
    )
    
    try:
        logger.info("Predictor running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        predictor.stop()

if __name__ == "__main__":
    main()
