import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import hashlib
import json
import requests
from web3 import Web3
import os
from typing import Dict, Optional
import logging
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CropMarketAnalyzer:
    def __init__(self, blockchain_node_url: Optional[str] = None):
        self.models = {}
        self.blockchain_connected = False
        self.web3 = None
        self.smart_contract = None
        
        if blockchain_node_url:
            self.connect_to_blockchain(blockchain_node_url)
    
    def connect_to_blockchain(self, node_url: str, contract_address: Optional[str] = None, contract_abi: Optional[str] = None):
        """Connect to Ethereum blockchain and optionally load a smart contract"""
        try:
            self.web3 = Web3(Web3.HTTPProvider(node_url))
            if self.web3.is_connected():
                self.blockchain_connected = True
                logger.info("Successfully connected to blockchain node")
                
                if contract_address and contract_abi:
                    self.smart_contract = self.web3.eth.contract(
                        address=contract_address,
                        abi=contract_abi
                    )
                    logger.info("Smart contract loaded successfully")
            else:
                logger.warning("Failed to connect to blockchain node")
        except Exception as e:
            logger.error(f"Error connecting to blockchain: {str(e)}")
    
    def get_realtime_market_data(self, crop: str, region: Optional[str] = None) -> Dict:
        """Get real-time market data for a specific crop"""
        api_data = self._get_market_data_from_apis(crop, region)
        
        blockchain_data = {}
        if self.blockchain_connected and self.smart_contract:
            try:
                blockchain_data = self._get_market_data_from_blockchain(crop)
            except Exception as e:
                logger.error(f"Error getting blockchain data: {str(e)}")
        
        market_data = {
            'crop': crop,
            'region': region,
            'timestamp': datetime.now().isoformat(),
            'price': api_data.get('price', 0),
            'demand': api_data.get('demand', 0),
            'supply': api_data.get('supply', 0),
            'price_trend': self._calculate_price_trend(api_data.get('historical_prices', [])),
            'demand_trend': self._calculate_demand_trend(api_data.get('historical_demand', [])),
            'supply_trend': self._calculate_supply_trend(api_data.get('historical_supply', [])),
            'blockchain_verified': blockchain_data.get('verified', False),
            'last_blockchain_update': blockchain_data.get('timestamp'),
            'supply_demand_ratio': self._calculate_supply_demand_ratio(
                api_data.get('demand', 0), 
                api_data.get('supply', 0)
            )
        }
        
        self._save_market_data(market_data)
        return market_data
    
    def _get_market_data_from_apis(self, crop: str, region: Optional[str] = None) -> Dict:
        """Simulate getting market data from agricultural APIs"""
        time.sleep(1)
        
        base_price = {
            'Wheat': 6.50,
            'Corn': 4.20,
            'Soybeans': 12.00,
            'Rice': 15.50,
            'Cotton': 0.85,
            'Potatoes': 150.00,
            'Tomatoes': 80.00,
            'Apples': 25.00,
            'Oranges': 12.00
        }.get(crop, 10.00)
        
        price_variation = np.random.uniform(-0.2, 0.2)
        if region:
            price_variation += np.random.uniform(-0.1, 0.1)
        
        current_price = base_price * (1 + price_variation)
        
        base_demand = np.random.randint(50, 200)
        demand = base_demand * (1 + np.random.uniform(-0.15, 0.15))
        
        base_supply = np.random.randint(40, 180)
        supply = base_supply * (1 + np.random.uniform(-0.2, 0.2))
        
        historical_prices = [current_price * (1 + np.random.uniform(-0.05, 0.05)) for _ in range(30)]
        historical_demand = [demand * (1 + np.random.uniform(-0.1, 0.1)) for _ in range(30)]
        historical_supply = [supply * (1 + np.random.uniform(-0.1, 0.1)) for _ in range(30)]
        
        return {
            'price': round(current_price, 2),
            'demand': round(demand),
            'supply': round(supply),
            'historical_prices': historical_prices,
            'historical_demand': historical_demand,
            'historical_supply': historical_supply
        }
    
    def _get_market_data_from_blockchain(self, crop: str) -> Dict:
        """Get market data from blockchain if available"""
        if not self.blockchain_connected or not self.smart_contract:
            return {}
            
        try:
            return {
                'verified': True,
                'timestamp': datetime.now().isoformat(),
                'price': None,
                'demand': None,
                'supply': None
            }
        except Exception as e:
            logger.error(f"Blockchain data retrieval failed: {str(e)}")
            return {}
    
    def _calculate_price_trend(self, historical_prices: list) -> float:
        if len(historical_prices) < 2:
            return 0.0
        
        current = historical_prices[-1]
        previous = historical_prices[0]
        return ((current - previous) / previous) * 100
    
    def _calculate_demand_trend(self, historical_demand: list) -> float:
        if len(historical_demand) < 2:
            return 0.0
        
        current = historical_demand[-1]
        previous = historical_demand[0]
        return ((current - previous) / previous) * 100
    
    def _calculate_supply_trend(self, historical_supply: list) -> float:
        if len(historical_supply) < 2:
            return 0.0
        
        current = historical_supply[-1]
        previous = historical_supply[0]
        return ((current - previous) / previous) * 100
    
    def _calculate_supply_demand_ratio(self, demand: float, supply: float) -> float:
        if demand == 0:
            return float('inf')
        return supply / demand
    
    def _save_market_data(self, market_data: Dict):
        try:
            os.makedirs('market_data', exist_ok=True)
            filename = f"market_data/{market_data['crop']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(market_data, f, indent=2)
            
            logger.info(f"Market data saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save market data: {str(e)}")
            return None
    
    def generate_market_report(self, crop: str, region: Optional[str] = None) -> str:
        market_data = self.get_realtime_market_data(crop, region)
        
        report = [
            f"MARKET REPORT FOR {crop.upper()}",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Region: {region if region else 'Global'}",
            "",
            f"Current Price: ${market_data['price']:.2f}",
            f"Price Trend: {'↑' if market_data['price_trend'] > 0 else '↓'} {abs(market_data['price_trend']):.1f}%",
            "",
            f"Current Demand: {market_data['demand']} units",
            f"Demand Trend: {'↑' if market_data['demand_trend'] > 0 else '↓'} {abs(market_data['demand_trend']):.1f}%",
            "",
            f"Current Supply: {market_data['supply']} units",
            f"Supply Trend: {'↑' if market_data['supply_trend'] > 0 else '↓'} {abs(market_data['supply_trend']):.1f}%",
            "",
            f"Supply-Demand Ratio: {market_data['supply_demand_ratio']:.2f}",
            f"Market Status: {'Oversupplied' if market_data['supply_demand_ratio'] > 1 else 'Undersupplied'}",
            "",
            f"Blockchain Verified: {'Yes' if market_data['blockchain_verified'] else 'No'}"
        ]
        
        return "\n".join(report)

# Update the if __name__ == "__main__": section in market.py with this:

if __name__ == "__main__":
    # Initialize analyzer with blockchain connection (optional)
    analyzer = CropMarketAnalyzer(
        blockchain_node_url=os.getenv('BLOCKCHAIN_NODE_URL')
    )
    
    # Run crop_api.py and get the recommended crop
    recommended_crop = "Wheat"  # Default fallback
    
    try:
        # Check if crop_api.py exists
        if not os.path.exists('crop_api.py'):
            logger.warning("crop_api.py not found in current directory")
            raise FileNotFoundError("crop_api.py not found")
        
        # Prepare environment for subprocess
        env = os.environ.copy()
        
        # Run crop_api.py in standalone mode with a timeout
        result = subprocess.run(
            ['python', 'crop_api.py', '--standalone'],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            env=env
        )
        
        # Check for errors in the subprocess
        if result.returncode != 0:
            logger.error(f"crop_api.py failed with error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, 'python crop_api.py')
        
        # Parse output for recommended crop
        for line in result.stdout.splitlines():
            if "Recommended Crop:" in line:
                recommended_crop = line.split("Recommended Crop:")[1].strip()
                break
            elif "PRIMARY RECOMMENDATION" in line:  # Alternative parsing
                parts = line.split()
                if len(parts) > 2:
                    recommended_crop = parts[1].strip('*').strip()
        
        logger.info(f"Recommended crop from crop_api.py: {recommended_crop}")
        
    except FileNotFoundError:
        logger.warning("Using default crop (Wheat) because crop_api.py is missing")
    except subprocess.TimeoutExpired:
        logger.error("crop_api.py timed out after 60 seconds")
        logger.info("Using default crop (Wheat)")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running crop_api.py: {e.stderr if e.stderr else 'Unknown error'}")
        logger.info("Using default crop (Wheat)")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.info("Using default crop (Wheat)")

    # Get real-time market data for the recommended crop
    print(f"\nAnalyzing market for recommended crop: {recommended_crop}")
    try:
        market_data = analyzer.get_realtime_market_data(recommended_crop)
        
        # Generate and print report
        report = analyzer.generate_market_report(recommended_crop)
        print("\n" + report)
        
        # Save detailed market data
        print(f"\nDetailed market data saved to JSON file")
    except Exception as e:
        logger.error(f"Failed to generate market report: {str(e)}")
        print("\nFailed to generate market report. Please check the logs.")