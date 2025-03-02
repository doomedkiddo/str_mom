import zmq
import json
import time
import os
import numpy as np
import pandas as pd
import threading
import lightgbm as lgb
from datetime import datetime, timedelta
import tempfile
import shutil
from loguru import logger

# Set up logger
logger.remove()
logger.add(lambda msg: print(msg), level="INFO")

class MockOrderBookPublisher:
    """A mock class that simulates publishing order book data via ZMQ"""
    
    def __init__(self, port=5557, delay=0.05):
        """Initialize ZMQ publisher"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        self.running = False
        self.delay = delay  # Delay between messages in seconds
        
    def start(self, duration=60):
        """Start publishing mock data for a given duration in seconds"""
        self.running = True
        self.thread = threading.Thread(target=self._publish_data, args=(duration,))
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Publisher started on port 5557, will run for {duration} seconds")
        
    def _publish_data(self, duration):
        """Publish mock order book data"""
        start_time = time.time()
        base_price = 50000.0  # Starting BTC price
        price_volatility = 10.0  # Price movement scale
        timestamp_ns = int(time.time() * 1e9)
        
        counter = 0
        
        while self.running and (time.time() - start_time) < duration:
            # Simulate price movement with random walk
            price_change = np.random.normal(0, price_volatility)
            base_price += price_change
            
            # Create mock order book with 5 levels
            timestamp_ns += int(self.delay * 1e9)
            mid_price = max(1.0, base_price)
            spread = mid_price * 0.0002  # 0.02% spread
            
            bids = []
            asks = []
            
            # Generate 5 bid levels
            for i in range(5):
                price = mid_price - spread/2 - i * spread
                size = 10.0 * (5-i) + np.random.random() * 5
                bids.append({"price": price, "quantity": size})
            
            # Generate 5 ask levels
            for i in range(5):
                price = mid_price + spread/2 + i * spread
                size = 10.0 * (5-i) + np.random.random() * 5
                asks.append({"price": price, "quantity": size})
            
            # Create and send message
            message = {
                "type": "depth",
                "symbol": "BTC-USDT-PERP",
                "timestamp": timestamp_ns,
                "bids": bids,
                "asks": asks
            }
            
            self.socket.send_string(json.dumps(message))
            counter += 1
            
            # Print progress every 100 messages
            if counter % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Published {counter} messages, elapsed time: {elapsed:.1f}s")
            
            time.sleep(self.delay)
        
        logger.info(f"Publisher finished after sending {counter} messages")
    
    def stop(self):
        """Stop the publisher"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.socket.close()
        self.context.term()
        logger.info("Publisher stopped")


def create_dummy_model(test_dir):
    """Create a dummy LightGBM model for testing"""
    # Create a very simple dataset
    X = np.random.random((100, 25200))  # 1800 * 14 features flattened
    y = np.random.randint(0, 3, 100)  # 3-class classification
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X, label=y)
    
    # Parameters for a simple model
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'num_leaves': 5,
        'learning_rate': 0.1,
        'n_estimators': 5
    }
    
    # Train a simple model
    model = lgb.train(params, train_data, num_boost_round=5)
    
    # Save model
    model_path = os.path.join(test_dir, "test_model.bin")
    model.save_model(model_path)
    
    return model_path


def test_predictor():
    """Test the OrderBookPredictor with simulated data"""
    # Import the OrderBookPredictor to test
    try:
        from predict import OrderBookPredictor
    except ImportError:
        logger.error("Could not import OrderBookPredictor from predict.py")
        return False
    
    # Create a temporary directory for test files
    test_dir = tempfile.mkdtemp()
    try:
        # Create a dummy model
        model_path = create_dummy_model(test_dir)
        
        # Start the publisher
        publisher = MockOrderBookPublisher(delay=0.01)  # Fast publishing for testing
        publisher.start(duration=30)  # Run for 30 seconds
        
        # Start the predictor with short save interval for testing
        predictor = OrderBookPredictor(
            model_path=model_path, 
            zmq_address="tcp://localhost:5557",
            data_save_interval=5,  # Save every 5 seconds
            confidence_threshold=0.3,  # Low threshold to get more signals for testing
            trading_symbol="DOGEUSDT",
            order_quantity_usdt=10.0
        )
        
        # Override data directory to use test directory
        predictor.data_dir = test_dir
        
        # Monitor signals during test
        signal_monitor_thread = threading.Thread(target=monitor_signals, args=(predictor,))
        signal_monitor_thread.daemon = True
        signal_monitor_thread.start()
        
        # Let it run for the test duration plus a small buffer
        try:
            time.sleep(35)  # Wait for test to complete
            
            # Check if data files were created
            data_files = [f for f in os.listdir(test_dir) if f.startswith("orderbook_data_")]
            if not data_files:
                logger.error("No data files were created!")
                return False
            
            logger.info(f"Found {len(data_files)} data files")
            
            # Check if at least one data file has content
            sample_file = os.path.join(test_dir, data_files[0])
            try:
                df = pd.read_feather(sample_file)
                logger.info(f"Sample data file has {len(df)} rows and {len(df.columns)} columns")
                
                # Check if key columns exist
                required_columns = ['mid_price', 'relative_spread', 'depth_imbalance']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"Data is missing required columns: {missing_columns}")
                    return False
                
                logger.info("Data validation passed")
            except Exception as e:
                logger.error(f"Failed to read data file: {e}")
                return False
            
            # Detailed summary of the trade signals
            display_trade_signals(predictor)
            
            # Check for active orders
            display_active_orders(predictor)
            
            return True
            
        finally:
            # Stop the predictor
            predictor.stop()
            
            # Stop the publisher
            publisher.stop()
        
    finally:
        # Clean up test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def monitor_signals(predictor):
    """Monitor and display signals as they happen"""
    last_signal_count = 0
    
    while True:
        current_count = len(predictor.trade_log)
        if current_count > last_signal_count:
            # New signals have been generated
            for i in range(last_signal_count, current_count):
                signal = predictor.trade_log[i]
                logger.info(f"🔔 NEW SIGNAL GENERATED: {signal['action']} at price {signal['price']:.8f}")
                
                # Check if order was placed
                if 'client_order_id' in signal:
                    logger.info(f"   ✅ Order placed with ID: {signal['client_order_id']}")
                    logger.info(f"   Symbol: {predictor.trading_symbol}, Confidence: {signal['confidence']:.4f}")
            
            last_signal_count = current_count
        
        time.sleep(0.5)  # Check every half second

def display_trade_signals(predictor):
    """Display detailed information about trade signals"""
    if not predictor.trade_log:
        logger.warning("❗ No trade signals were generated during the test")
    else:
        logger.info("\n" + "="*50)
        logger.info(f"📊 TRADING SIGNALS SUMMARY (Total: {len(predictor.trade_log)})")
        logger.info("="*50)
        
        # Count actions by type
        action_counts = {}
        for signal in predictor.trade_log:
            action = signal['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        logger.info(f"Signal distribution: {action_counts}")
        logger.info("\nDetailed signal list:")
        
        # Display each signal
        for i, signal in enumerate(predictor.trade_log):
            logger.info(f"\nSignal #{i+1}:")
            logger.info(f"  Timestamp: {signal['timestamp']}")
            logger.info(f"  Action: {signal['action']}")
            logger.info(f"  Price: {signal['price']:.8f}")
            logger.info(f"  Confidence: {signal['confidence']:.4f}")
            
            # Show prediction details if available
            if 'prediction' in signal and signal['prediction']:
                pred = signal['prediction']
                if isinstance(pred, dict):
                    if 'class' in pred:
                        logger.info(f"  Predicted class: {pred['class']}")
                        if 'probabilities' in pred:
                            probs = pred['probabilities']
                            logger.info(f"  Class probabilities: Down={probs.get('down', 0):.4f}, "
                                        f"Neutral={probs.get('neutral', 0):.4f}, "
                                        f"Up={probs.get('up', 0):.4f}")
                    elif 'predicted_return' in pred:
                        logger.info(f"  Predicted return: {pred['predicted_return']:.4f}")
            
            # Show order ID if available
            if 'client_order_id' in signal:
                logger.info(f"  Order ID: {signal['client_order_id']}")

def display_active_orders(predictor):
    """Display information about active orders at the end of the test"""
    if not predictor.active_orders:
        logger.info("\n❗ No active orders at the end of test")
    else:
        logger.info("\n" + "="*50)
        logger.info(f"📋 ACTIVE ORDERS (Total: {len(predictor.active_orders)})")
        logger.info("="*50)
        
        for client_id, order in predictor.active_orders.items():
            logger.info(f"\nOrder ID: {client_id}")
            logger.info(f"  Symbol: {order['symbol']}")
            logger.info(f"  Side: {order['side']}")
            logger.info(f"  Price: {order['price']:.8f}")
            logger.info(f"  Quantity: {order['quantity']:.8f}")
            logger.info(f"  TP/SL: {'Yes' if order.get('with_tp_sl', False) else 'No'}")
            logger.info(f"  Timestamp: {order['timestamp']}")

if __name__ == "__main__":
    logger.info("Starting OrderBookPredictor test...")
    success = test_predictor()
    if success:
        logger.info("✅ Test completed successfully! The predictor is working correctly.")
    else:
        logger.error("❌ Test failed. Please check the errors above.") 
