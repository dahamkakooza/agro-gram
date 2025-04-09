import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
import yfinance as yf
from geopy.geocoders import Nominatim
import requests
import json
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from functools import lru_cache
import time
from enum import Enum, auto
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
# At the top of crop_api.py (replace the Django settings import)
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_recommender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
pd.set_option('future.no_silent_downcasting', True)

class WeatherRiskLevel(Enum):
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    EXTREME = auto()

    def __str__(self):
        return self.name

class AgricultureChatAssistant:
    """AI assistant for answering agriculture-related questions using Google Gemini API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key not configured")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.base_prompt = """You are an expert agricultural assistant specialized in crop cultivation, 
        market prices, fertilizers, and farming techniques. Provide accurate, practical advice 
        tailored to small and large-scale farmers. Be concise but thorough in your explanations.
        
        When asked about:
        - Crop prices: Provide current market trends and factors affecting prices
        - Fertilizers: Recommend appropriate types and application methods
        - Cultivation: Offer best practices for soil preparation, planting, and care
        - Pests/Diseases: Suggest organic and chemical control methods
        - Any other agriculture topic: Provide expert guidance
        
        Always remind farmers to consult local agricultural extension services for specific regional advice.
        Important: For crop prices, also check the CROP_MARKET_DATA in the system for current values.
        """
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def ask_question(self, question: str, max_tokens: int = 500) -> str:
        """Ask an agriculture-related question to the AI assistant"""
        try:
            response = self.model.generate_content(
                f"{self.base_prompt}\n\nQuestion: {question}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return "I'm sorry, I couldn't process your question at this time. Please try again later."

class ProfessionalCropRecommender:
    """Advanced crop recommendation system with weather, market intelligence, and AI chat"""
    
    VALID_SOIL_TYPES = ['Loam', 'Clay', 'Sandy', 'SiltyClay', 'ClayLoam', 'SandyLoam', 'Peat', 'SiltLoam']
    CACHE_DIR = "cache"
    PRICES_FILE = "crop_prices.json"
    SEASONAL_CACHE_DAYS = 30
    REQUEST_DELAY = 1.0  # Seconds between API calls
    MIN_CONFIDENCE_THRESHOLD = 0.05  # Minimum confidence to show recommendation
    
    CROP_MARKET_DATA = {
        'Wheat': {'ticker': 'KE=F', 'profit_per_acre': 400, 'unit': 'bushel', 'fallback_price': 6.50},
        'Corn': {'ticker': 'ZC=F', 'profit_per_acre': 600, 'unit': 'bushel', 'fallback_price': 4.20},
        'Soybeans': {'ticker': 'ZS=F', 'profit_per_acre': 500, 'unit': 'bushel', 'fallback_price': 12.00},
        'Rice': {'ticker': 'ZR=F', 'profit_per_acre': 450, 'unit': 'hundredweight', 'fallback_price': 15.50},
        'Cotton': {'ticker': 'CT=F', 'profit_per_acre': 550, 'unit': 'pound', 'fallback_price': 0.85},
        'Potatoes': {'profit_per_acre': 800, 'unit': 'ton', 'fallback_price': 150.00},
        'Tomatoes': {'profit_per_acre': 1200, 'unit': 'ton', 'fallback_price': 80.00},
        'Apples': {'profit_per_acre': 1000, 'unit': 'bushel', 'fallback_price': 25.00},
        'Oranges': {'profit_per_acre': 950, 'unit': 'box', 'fallback_price': 12.00}
    }
    
    def __init__(self):
        self.model = None
        self.df = None
        self.preprocessor = None
        self.geolocator = Nominatim(user_agent="crop_recommender_pro_v6", timeout=10)
        self.price_db = {}
        self.weather_cache = {}
        self.chat_assistant = AgricultureChatAssistant(GEMINI_API_KEY)
        self._setup_cache()
        self._load_price_database()

    def chat_with_assistant(self, question: str) -> str:
        """Interact with the agriculture expert assistant"""
        if not question.strip():
            return "Please ask a question about crops, fertilizers, or farming practices."
        
        logger.info(f"User question: {question}")
        response = self.chat_assistant.ask_question(question)
        logger.info(f"AI response: {response[:200]}...")  # Log first 200 chars
        
        return response

    def generate_report(self, suggestions: List[dict], model_info: dict, 
                       user_input: dict, location: Optional[str] = None) -> str:
        """Generate a detailed recommendation report with explanations and alternatives"""
        if not suggestions:
            return "No suitable crops found for your conditions."
            
        report = []
        report.append("="*80)
        report.append("🌱 PROFESSIONAL CROP RECOMMENDATION REPORT 🌱".center(80))
        report.append("="*80)
        report.append(f"\n📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if location:
            report.append(f"\n📍 Location: {location}")
        
        # User conditions section
        report.append("\n" + "📌 YOUR GROWING CONDITIONS".center(80, '-'))
        report.append(f"\n🌡️ Soil Temperature: {user_input['soil_temp']:.1f}°C")
        report.append(f"🧪 Soil pH: {user_input['soil_ph']:.1f}")
        report.append(f"🏜️ Soil Type: {user_input['soil_type']}")
        report.append(f"💧 Annual Rainfall: {user_input['rainfall']} mm")
        report.append(f"💦 Humidity: {user_input['humidity']}%")
        
        # Model information
        report.append("\n" + "📊 MODEL ANALYSIS".center(80, '-'))
        report.append(f"\n🔍 Model Accuracy: {model_info['best_score']:.1%}")
        if model_info['best_score'] < 0.5:
            report.append("⚠️ Warning: Model accuracy is below 50%. Recommendations may not be reliable.")
        report.append(f"⚙️ Best Parameters: {model_info['best_params']}")
        
        # Filter suggestions by confidence threshold
        filtered_suggestions = [s for s in suggestions if s['confidence'] >= self.MIN_CONFIDENCE_THRESHOLD]
        
        if not filtered_suggestions:
            report.append("\n⚠️ No crops met the minimum confidence threshold. Showing closest matches anyway:")
            filtered_suggestions = suggestions[:3]  # Show top 3 regardless of confidence
        
        # Primary recommendation
        if len(filtered_suggestions) > 0:
            primary = filtered_suggestions[0]
            primary_data = self.df[self.df['Crop'] == primary['crop']].iloc[0]
            
            report.append("\n" + "⭐ PRIMARY RECOMMENDATION ⭐".center(80, ' '))
            report.append(f"\n1. {primary['crop'].upper()} ({primary['suitability']})")
            report.append("-"*60)
            report.append(f"✅ Confidence: {primary['confidence']:.1%}")
            if primary['confidence'] < 0.2:
                report.append("⚠️ Note: Low confidence recommendation. Verify suitability with local experts.")
            
            # Parameter comparison
            report.append("\n" + "🔬 IDEAL vs YOUR CONDITIONS".center(60, '-'))
            report.append(f"⟡ Soil pH:    {primary_data['Soil pH Min']:.1f}-{primary_data['Soil pH Max']:.1f} (Ideal) vs {user_input['soil_ph']:.1f} (Yours)")
            report.append(f"⟡ Temperature: {primary_data['Soil Temp Min (°C)']}°C-{primary_data['Soil Temp Max (°C)']}°C vs {user_input['soil_temp']:.1f}°C")
            report.append(f"⟡ Rainfall:   {primary_data['Rainfall Min (mm/year)']}-{primary_data['Rainfall Max (mm/year)']}mm vs {user_input['rainfall']}mm")
            report.append(f"⟡ Humidity:   {primary_data['Humidity Min (%)']}%-{primary_data['Humidity Max (%)']}% vs {user_input['humidity']}%")
            
            # Weather analysis
            if 'weather_data' in primary:
                report.append("\n" + "🌦️ CURRENT WEATHER CONDITIONS".center(60, '-'))
                report.append(f"🌡️ Temperature: {primary['weather_data']['temperature']}°C")
                report.append(f"💧 Precipitation: {primary['weather_data']['precipitation']} mm")
                report.append(f"☁️ Cloud Cover: {primary['weather_data'].get('cloud_cover', 'N/A')}%")
                report.append(f"💨 Wind Speed: {primary['weather_data'].get('wind_speed', 'N/A')} km/h")
            
            # Market analysis
            report.append("\n" + "💲 MARKET ANALYSIS".center(60, '-'))
            report.append(f"💰 Current Price: {primary['market_price'] if primary['market_price'] else 'N/A'} {primary['price_unit']}")
            report.append(f"📈 Price Trend: {'↑' if primary['price_trend'] > 0 else '↓'} {abs(primary['price_trend']):.1f}%")
            report.append(f"💵 Base Profit: ${primary['base_profit']:,.0f}/acre")
            report.append(f"📊 Adjusted Profit: ${primary['adjusted_profit']:,.0f}/acre")
            
            # Risk analysis
            report.append("\n" + "⚠️ RISK ASSESSMENT".center(60, '-'))
            report.append(f"🌦️ Weather Risk: {primary['weather_risk']}")
            report.append(f"📉 Market Risk: {primary['market_risk']:.1f}/1.0")
            report.append(f"☠️ Combined Risk: {primary['combined_risk']:.1f}/1.0")
        
        # Alternative options
        if len(filtered_suggestions) > 1:
            report.append("\n" + "🔀 ALTERNATIVE OPTIONS".center(80, '~'))
            
            for i in range(1, min(3, len(filtered_suggestions))):  # Show up to 2 alternatives
                alt = filtered_suggestions[i]
                alt_data = self.df[self.df['Crop'] == alt['crop']].iloc[0]
                
                report.append(f"\n{i+1}. {alt['crop'].upper()} ({alt['suitability']})")
                report.append("-"*60)
                report.append(f"✅ Confidence: {alt['confidence']:.1%}")
                
                # Key differences
                report.append("\n" + "🔄 COMPARED TO PRIMARY".center(60, '-'))
                if alt['adjusted_profit'] > primary['adjusted_profit']:
                    report.append(f"💰 Higher profit potential by ${alt['adjusted_profit'] - primary['adjusted_profit']:,.0f}/acre")
                else:
                    report.append(f"💸 Lower profit by ${primary['adjusted_profit'] - alt['adjusted_profit']:,.0f}/acre")
                
                if alt['combined_risk'] < primary['combined_risk']:
                    report.append(f"🛡️ Lower risk by {primary['combined_risk'] - alt['combined_risk']:.1f} points")
                else:
                    report.append(f"⚡ Higher risk by {alt['combined_risk'] - primary['combined_risk']:.1f} points")
                
                # Weather data for alternatives
                if 'weather_data' in alt:
                    report.append("\n" + "⚡ CURRENT WEATHER".center(60, '-'))
                    report.append(f"🌡️ Temp: {alt['weather_data']['temperature']}°C")
                    report.append(f"💧 Precip: {alt['weather_data']['precipitation']} mm")
                
                # Quick stats
                report.append("\n" + "⚡ QUICK STATS".center(60, '-'))
                report.append(f"🌱 Ideal pH: {alt_data['Soil pH Min']:.1f}-{alt_data['Soil pH Max']:.1f}")
                report.append(f"🌡️ Temp Range: {alt_data['Soil Temp Min (°C)']}°C-{alt_data['Soil Temp Max (°C)']}°C")
                report.append(f"💧 Water Needs: {alt_data['Rainfall Min (mm/year)']}-{alt_data['Rainfall Max (mm/year)']}mm/yr")
        
        # Final advice
        report.append("\n" + "📝 FARMING STRATEGY ADVICE".center(80, '='))
        if len(filtered_suggestions) >= 3:
            report.append(f"\nConsider growing {filtered_suggestions[0]['crop']} as your main crop with:")
            report.append(f"- {filtered_suggestions[1]['crop']} (better for {'pH' if abs(user_input['soil_ph'] - (filtered_suggestions[1]['soil_ph'])) < abs(user_input['soil_ph'] - (filtered_suggestions[0]['soil_ph'])) else 'temperature'} conditions)")
            report.append(f"- {filtered_suggestions[2]['crop']} (lower risk option)")
        elif len(filtered_suggestions) > 0:
            report.append(f"\nConsider growing {filtered_suggestions[0]['crop']} with local guidance due to marginal conditions")
        
        # Add AI advice section
        report.append("\n" + "🤖 AI AGRICULTURE ASSISTANT".center(80, '='))
        report.append("\nRemember you can ask me questions anytime about:")
        report.append("- Current crop prices and market trends")
        report.append("- Fertilizer recommendations")
        report.append("- Pest and disease control")
        report.append("- Best cultivation practices")
        report.append("- Any other farming questions")
        report.append("\n" + "="*80)
        
        return "\n".join(report)

    def save_report_to_file(self, report_text: str, filename: str = "crop_recommendations.txt"):
        """Save the recommendation report to a text file with emoji handling"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            return filename
        except Exception as e:
            logger.error(f"Failed to save text report: {str(e)}")
            return None

    def save_report_as_html(self, report_text: str, filename: str = "crop_recommendations.html"):
        """Save report as HTML with emoji support"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Crop Recommendations</title>
                <style>
                    body {{ 
                        font-family: Arial; 
                        line-height: 1.6;
                        max-width: 900px;
                        margin: 0 auto;
                        padding: 20px;
                        color: #333;
                    }}
                    .header {{ 
                        text-align: center; 
                        margin-bottom: 20px;
                    }}
                    .section {{ 
                        margin-bottom: 25px;
                        padding: 15px;
                        background-color: #f9f9f9;
                        border-radius: 5px;
                    }}
                    .primary {{
                        background-color: #e8f5e9;
                        border-left: 5px solid #4caf50;
                    }}
                    .alternatives {{
                        background-color: #e3f2fd;
                        border-left: 5px solid #2196f3;
                    }}
                    .advice {{
                        background-color: #fff8e1;
                        border-left: 5px solid #ffc107;
                    }}
                    .warning {{
                        background-color: #ffebee;
                        border-left: 5px solid #f44336;
                        padding: 10px;
                        margin: 10px 0;
                    }}
                    .crop-name {{
                        font-weight: bold;
                        color: #2e7d32;
                        font-size: 1.1em;
                    }}
                    .comparison {{
                        font-family: monospace;
                        background-color: #f5f5f5;
                        padding: 10px;
                        border-radius: 3px;
                    }}
                    .risk-high {{ color: #e53935; }}
                    .risk-medium {{ color: #fb8c00; }}
                    .risk-low {{ color: #43a047; }}
                    .ai-assistant {{
                        background-color: #f3e5f5;
                        border-left: 5px solid #9c27b0;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🌱 Professional Crop Recommendation Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <pre>{report_text}</pre>
            </body>
            </html>
            """
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return filename
        except Exception as e:
            logger.error(f"Failed to save HTML report: {str(e)}")
            return None

    def plot_recommendations(self, suggestions: List[dict]) -> Optional[str]:
        """Create a visualization of the recommendations"""
        if not suggestions:
            return None
            
        try:
            # Filter suggestions by confidence threshold
            filtered_suggestions = [s for s in suggestions if s['confidence'] >= self.MIN_CONFIDENCE_THRESHOLD]
            if not filtered_suggestions:
                filtered_suggestions = suggestions[:3]  # Show top 3 if none meet threshold
            
            # Prepare data
            crops = [s['crop'] for s in filtered_suggestions]
            profits = [s['adjusted_profit'] for s in filtered_suggestions]
            confidences = [s['confidence'] * 100 for s in filtered_suggestions]
            risks = [s['combined_risk'] * 100 for s in filtered_suggestions]
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Profit bar plot
            sns.barplot(x=profits, y=crops, ax=ax1, hue=crops, palette='viridis', legend=False)
            ax1.set_title('Adjusted Profit per Acre')
            ax1.set_xlabel('USD per Acre')
            ax1.bar_label(ax1.containers[0], fmt='$%.0f')
            
            # Confidence and risk scatter plot
            ax2.scatter(confidences, crops, color='green', label='Confidence', s=100)
            ax2.scatter(risks, crops, color='red', label='Risk', s=100)
            ax2.set_title('Confidence vs Risk')
            ax2.set_xlabel('Percentage')
            ax2.legend()
            
            # Save plot
            plt.tight_layout()
            plot_path = "crop_recommendations.png"
            plt.savefig(plot_path)
            plt.close()
            
            return plot_path
        except Exception as e:
            logger.error(f"Failed to create plot: {str(e)}")
            return None

    def debug_conditions(self, user_input: dict):
        """Show debugging information for user conditions"""
        print("\nDEBUG INFORMATION:")
        print("="*50)
        
        # Find closest matches for each parameter
        for param, value in user_input.items():
            if param == 'soil_type':
                continue
                
            print(f"\nClosest crops for {param} = {value}:")
            
            # Find crops with closest parameter ranges
            if param in ['soil_ph', 'soil_temp']:
                self.df['distance'] = abs((self.df[f'{param.capitalize()} Min'] + self.df[f'{param.capitalize()} Max']) / 2 - value)
            else:
                self.df['distance'] = abs((self.df[f'{param.capitalize()} Min (mm/year)'] + self.df[f'{param.capitalize()} Max (mm/year)']) / 2 - value)
            
            closest = self.df.nsmallest(3, 'distance')[['Crop', 'distance']]
            for _, row in closest.iterrows():
                print(f" - {row['Crop']} (distance: {row['distance']:.1f})")

    def load_and_merge_data(self):
        """Load and merge crop data from multiple sources"""
        try:
            # Load basic crop data
            basic_df = pd.read_csv('crop.csv')
        
            # Load professional crop data if available
            if os.path.exists('professional_crops.csv'):
                pro_df = pd.read_csv('professional_crops.csv')
                self.df = pd.concat([basic_df, pro_df], ignore_index=True)
            else:
                self.df = basic_df
            
            # Clean and preprocess data
            self.df = self.df.dropna(subset=['Crop'])
        
            # Remove crops with insufficient data
            crop_counts = self.df['Crop'].value_counts()
            valid_crops = crop_counts[crop_counts >= 3].index  # Increased minimum samples
            self.df = self.df[self.df['Crop'].isin(valid_crops)]
        
            logger.info(f"Loaded data for {len(self.df)} crop varieties")
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return False

    def validate_inputs(self, soil_ph: float, soil_temp: float, soil_type: str, 
                      rainfall: int, humidity: float):
        """Validate user inputs"""
        if not (3.5 <= soil_ph <= 9.5):
            raise ValueError("Soil pH must be between 3.5 and 9.5")
        if soil_temp < -20 or soil_temp > 50:
            raise ValueError("Soil temperature must be between -20°C and 50°C")
        if soil_type not in self.VALID_SOIL_TYPES:
            raise ValueError(f"Invalid soil type. Must be one of: {', '.join(self.VALID_SOIL_TYPES)}")
        if rainfall < 0 or rainfall > 5000:
            raise ValueError("Annual rainfall must be between 0 and 5000 mm")
        if not (0 <= humidity <= 100):
            raise ValueError("Humidity must be between 0% and 100%")

    def train_model(self) -> Optional[dict]:
        """Train the recommendation model with hyperparameter tuning"""
        if self.df is None:
            logger.error("No data loaded for training")
            return None
            
        try:
            # Prepare features and target
            X = self.df.drop(columns=['Crop'])
            y = self.df['Crop']
            
            # Check class distribution
            class_counts = y.value_counts()
            min_samples = class_counts.min()
            n_splits = min(5, min_samples)  # Don't exceed minimum class count
            
            if n_splits < 2:
                raise ValueError(f"Some crops have only {min_samples} samples - need at least 2 for cross-validation")
                
            logger.info(f"Using {n_splits}-fold CV (minimum class count: {min_samples})")
            
            # Define preprocessing
            numeric_features = ['Soil pH Min', 'Soil pH Max', 
                              'Soil Temp Min (°C)', 'Soil Temp Max (°C)',
                              'Rainfall Min (mm/year)', 'Rainfall Max (mm/year)',
                              'Humidity Min (%)', 'Humidity Max (%)']
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))
            ])
            
            categorical_features = ['Soil Type']
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Define model pipeline with alternative algorithms
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
            
            # Expanded parameter grid
            param_grid = [
                {
                    'classifier': [RandomForestClassifier(random_state=42)],
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__class_weight': [None, 'balanced']
                },
                {
                    'classifier': [GradientBoostingClassifier(random_state=42)],
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.1, 0.05],
                    'classifier__max_depth': [3, 5]
                }
            ]
            
            # Train with cross-validation
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=cv, 
                scoring='accuracy', 
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            
            return {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'n_splits_used': n_splits,
                'min_class_count': min_samples
            }
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_open_meteo_data(self, latitude: float, longitude: float) -> dict:
        """Get comprehensive weather data from Open-Meteo API"""
        try:
            time.sleep(self.REQUEST_DELAY)
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,precipitation,cloud_cover,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_hours",
                "hourly": "relative_humidity_2m",
                "timezone": "auto"
            }
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Open-Meteo API request failed: {str(e)}")
            raise

    def _get_weather_analysis(self, crop: str, location: str) -> dict:
        """Get comprehensive weather analysis for a crop at a location"""
        try:
            # Get coordinates from location
            geo = self.geolocator.geocode(location)
            if not geo:
                logger.warning(f"Could not geocode location: {location}")
                return None
            
            # Get weather data from Open-Meteo
            weather_data = self._get_open_meteo_data(geo.latitude, geo.longitude)
            
            # Extract current weather data
            current = weather_data.get('current', {})
            daily = weather_data.get('daily', {})
            hourly = weather_data.get('hourly', {})
            
            # Calculate averages
            avg_humidity = np.mean(hourly.get('relative_humidity_2m', [0]))
            avg_precip_hours = np.mean(daily.get('precipitation_hours', [0]))
            
            return {
                'temperature': current.get('temperature_2m', 0),
                'precipitation': current.get('precipitation', 0),
                'cloud_cover': current.get('cloud_cover', 0),
                'wind_speed': current.get('wind_speed_10m', 0),
                'avg_humidity': avg_humidity,
                'avg_precip_hours': avg_precip_hours,
                'max_temp': np.max(daily.get('temperature_2m_max', [0])),
                'min_temp': np.min(daily.get('temperature_2m_min', [0]))
            }
        except Exception as e:
            logger.warning(f"Failed to get weather analysis: {str(e)}")
            return None

    def generate_suggestions(self, user_input: dict, location: Optional[str] = None) -> List[dict]:
        """Generate crop recommendations with market and weather analysis"""
        if self.model is None:
            logger.error("Model not trained")
            return []
            
        try:
            # Prepare input data
            input_df = pd.DataFrame([{
                'Soil pH Min': user_input['soil_ph'],
                'Soil pH Max': user_input['soil_ph'],
                'Soil Temp Min (°C)': user_input['soil_temp'],
                'Soil Temp Max (°C)': user_input['soil_temp'],
                'Rainfall Min (mm/year)': user_input['rainfall'],
                'Rainfall Max (mm/year)': user_input['rainfall'],
                'Humidity Min (%)': user_input['humidity'],
                'Humidity Max (%)': user_input['humidity'],
                'Soil Type': user_input['soil_type']
            }])
            
            # Get predictions
            probas = self.model.predict_proba(input_df)[0]
            classes = self.model.named_steps['classifier'].classes_
            
            # Get weather data if location is provided
            weather_data = None
            if location:
                weather_data = self._get_weather_analysis(classes[0], location)  # Get for first crop
            
            # Combine with crop data
            suggestions = []
            for crop, prob in zip(classes, probas):
                crop_data = self.df[self.df['Crop'] == crop].iloc[0]
                
                # Get market data
                market_data = self._get_market_data(crop)
                
                # Get weather risk with fallback
                weather_risk = WeatherRiskLevel.MODERATE  # Default fallback
                if location:
                    try:
                        weather_risk = self._get_weather_risk(crop, location)
                    except Exception as e:
                        logger.warning(f"Using fallback weather risk due to error: {str(e)}")
                
                # Calculate adjusted profit
                base_profit = market_data.get('profit_per_acre', 0)
                adjusted_profit = base_profit * (1 + (prob - 0.5))  # Scale by confidence
                
                # Risk assessment
                market_risk = 1 - (market_data.get('price_trend', 0) / 100 if 'price_trend' in market_data else 0.5)
                combined_risk = (weather_risk.value + market_risk) / 2
                
                suggestion = {
                    'crop': crop,
                    'confidence': prob,
                    'suitability': self._get_suitability_label(prob),
                    'market_price': market_data.get('current_price'),
                    'price_unit': market_data.get('unit', ''),
                    'price_trend': market_data.get('price_trend', 0),
                    'base_profit': base_profit,
                    'adjusted_profit': adjusted_profit,
                    'weather_risk': weather_risk,
                    'market_risk': market_risk,
                    'combined_risk': combined_risk,
                    'soil_ph': (crop_data['Soil pH Min'] + crop_data['Soil pH Max']) / 2,
                    'soil_temp': (crop_data['Soil Temp Min (°C)'] + crop_data['Soil Temp Max (°C)']) / 2,
                    'rainfall': (crop_data['Rainfall Min (mm/year)'] + crop_data['Rainfall Max (mm/year)']) / 2,
                    'humidity': (crop_data['Humidity Min (%)'] + crop_data['Humidity Max (%)']) / 2
                }
                
                # Add weather data if available
                if weather_data:
                    suggestion['weather_data'] = weather_data
                
                suggestions.append(suggestion)
            
            # Sort by adjusted profit and confidence
            suggestions.sort(key=lambda x: (-x['adjusted_profit'], -x['confidence']))
            return suggestions[:10]  # Return more options for filtering
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {str(e)}")
            return []

    def _get_suitability_label(self, probability: float) -> str:
        """Get human-readable suitability label"""
        if probability >= 0.9:
            return "Excellent Match"
        elif probability >= 0.7:
            return "Very Good Match"
        elif probability >= 0.5:
            return "Good Match"
        elif probability >= 0.3:
            return "Moderate Match"
        elif probability >= 0.1:
            return "Marginal Match"
        else:
            return "Poor Match"

    def _get_weather_risk(self, crop: str, location: str) -> WeatherRiskLevel:
        """Get weather risk assessment for a location using Open-Meteo"""
        cache_key = f"weather_{location}_{crop}"
        cached = self._load_cache('weather', cache_key)
        if cached:
            return WeatherRiskLevel[cached['risk_level']]
            
        try:
            # Get coordinates from location
            geo = self.geolocator.geocode(location)
            if not geo:
                logger.warning(f"Could not geocode location: {location}")
                return WeatherRiskLevel.MODERATE
            
            # Get comprehensive weather data
            weather_data = self._get_open_meteo_data(geo.latitude, geo.longitude)
            
            # Extract relevant weather data
            current_temp = weather_data.get('current', {}).get('temperature_2m', 20)
            current_precip = weather_data.get('current', {}).get('precipitation', 0)
            wind_speed = weather_data.get('current', {}).get('wind_speed_10m', 0)
            
            # Get daily data for more comprehensive analysis
            daily_data = weather_data.get('daily', {})
            if daily_data:
                avg_max_temp = np.mean(daily_data.get('temperature_2m_max', [current_temp]))
                avg_min_temp = np.mean(daily_data.get('temperature_2m_min', [current_temp]))
                avg_precip = np.mean(daily_data.get('precipitation_sum', [current_precip]))
            else:
                avg_max_temp = current_temp
                avg_min_temp = current_temp
                avg_precip = current_precip
            
            # Get crop requirements
            crop_data = self.df[self.df['Crop'] == crop].iloc[0]
            temp_min = crop_data['Soil Temp Min (°C)']
            temp_max = crop_data['Soil Temp Max (°C)']
            rainfall_min = crop_data['Rainfall Min (mm/year)'] / 365  # Convert to daily
            rainfall_max = crop_data['Rainfall Max (mm/year)'] / 365
            
            # Calculate temperature risk using average of max and min temps
            avg_temp = (avg_max_temp + avg_min_temp) / 2
            temp_risk = 0
            if avg_temp < temp_min:
                temp_risk = (temp_min - avg_temp) / 5  # 5°C below is high risk
            elif avg_temp > temp_max:
                temp_risk = (avg_temp - temp_max) / 5  # 5°C above is high risk
                
            # Calculate precipitation risk
            precip_risk = 0
            if avg_precip < rainfall_min:
                precip_risk = (rainfall_min - avg_precip) / rainfall_min if rainfall_min > 0 else 0
            elif avg_precip > rainfall_max:
                precip_risk = (avg_precip - rainfall_max) / rainfall_max if rainfall_max > 0 else 0
                
            # Calculate wind risk (high winds can damage crops)
            wind_risk = min(wind_speed / 20, 1)  # 20 km/h is considered high wind
                
            combined_risk = (temp_risk + precip_risk + wind_risk) / 3
            
            # Determine risk level
            if combined_risk > 0.7:
                risk_level = WeatherRiskLevel.EXTREME
            elif combined_risk > 0.5:
                risk_level = WeatherRiskLevel.HIGH
            elif combined_risk > 0.3:
                risk_level = WeatherRiskLevel.MODERATE
            else:
                risk_level = WeatherRiskLevel.LOW
                
            # Cache the result
            self._save_cache({
                'risk_level': risk_level.name,
                'last_updated': datetime.now().isoformat(),
                'weather_data': {
                    'temperature': avg_temp,
                    'precipitation': avg_precip,
                    'wind_speed': wind_speed
                }
            }, 'weather', cache_key)
            
            return risk_level
            
        except Exception as e:
            logger.warning(f"Using fallback weather risk due to error: {str(e)}")
            return WeatherRiskLevel.MODERATE

    def _get_market_data(self, crop: str) -> dict:
        """Get current market data for a crop"""
        if crop not in self.CROP_MARKET_DATA:
            return {
                'current_price': None,
                'price_trend': 0,
                'profit_per_acre': 0,
                'unit': ''
            }
            
        # Check cache first
        cache_key = f"market_{crop}"
        cached = self._load_cache('market', cache_key)
        if cached:
            return cached
            
        # Get fresh data
        market_info = self.CROP_MARKET_DATA[crop]
        result = {
            'profit_per_acre': market_info['profit_per_acre'],
            'unit': market_info['unit'],
            'current_price': market_info['fallback_price'],
            'price_trend': 0
        }
        
        # Try to get live data for crops with tickers
        if 'ticker' in market_info:
            try:
                ticker = market_info['ticker']
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1y')
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[0]
                    trend = ((current - prev) / prev) * 100
                    
                    result['current_price'] = round(current, 2)
                    result['price_trend'] = round(trend, 1)
            except Exception as e:
                logger.warning(f"Failed to get market data for {crop}: {str(e)}")
        
        # Update cache
        result['last_updated'] = datetime.now().isoformat()
        self._save_cache(result, 'market', cache_key)
        
        return result

    def _setup_cache(self):
        """Initialize cache directory with cleanup"""
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            # Cleanup old cache files
            for f in os.listdir(self.CACHE_DIR):
                if f.endswith('.json'):
                    filepath = os.path.join(self.CACHE_DIR, f)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        if (datetime.now() - file_time) > timedelta(days=7):
                            os.remove(filepath)
                    except (OSError, ValueError) as e:
                        logger.warning(f"Failed to process cache file {f}: {str(e)}")
        except OSError as e:
            logger.error(f"Failed to setup cache directory: {str(e)}")

    def _load_cache(self, cache_type: str, key: str) -> Optional[dict]:
        """Load cached data if available and recent"""
        cache_file = os.path.join(self.CACHE_DIR, f"{cache_type}_{key}.json")
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Handle timezone-aware and naive datetimes
            last_updated = data.get('last_updated', '1970-01-01')
            if 'Z' in last_updated or '+' in last_updated:
                # Timezone-aware datetime
                cache_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            else:
                # Naive datetime
                cache_time = datetime.fromisoformat(last_updated)
            
            # Make both datetimes naive for comparison
            cache_time = cache_time.replace(tzinfo=None)
            current_time = datetime.now().replace(tzinfo=None)
        
            if (current_time - cache_time) > timedelta(days=self.SEASONAL_CACHE_DAYS):
                return None
            
            return data
        except (json.JSONDecodeError, IOError, ValueError) as e:
            logger.warning(f"Cache load error: {str(e)}")
            return None

    def _save_cache(self, data: dict, cache_type: str, key: str):
        """Save data to cache with proper datetime handling"""
        try:
            # Ensure datetime is saved in ISO format without timezone
            if 'last_updated' in data:
                if isinstance(data['last_updated'], datetime):
                    data['last_updated'] = data['last_updated'].isoformat()
                elif 'Z' in data['last_updated']:
                    # Remove timezone info if present
                    data['last_updated'] = data['last_updated'].split('Z')[0]
                
            cache_file = os.path.join(self.CACHE_DIR, f"{cache_type}_{key}.json")
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except (IOError, TypeError) as e:
            logger.warning(f"Failed to save cache: {str(e)}")

    def _load_price_database(self):
        """Load price database from file"""
        try:
            if os.path.exists(self.PRICES_FILE):
                with open(self.PRICES_FILE, 'r') as f:
                    self.price_db = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load price database: {str(e)}")
            self.price_db = {}

    def _save_price_database(self):
        """Save price database to file"""
        try:
            with open(self.PRICES_FILE, 'w') as f:
                json.dump(self.price_db, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save price database: {str(e)}")

def main(standalone=False):
    """Enhanced main function with interactive menu"""
    if standalone:
        # Simplified version for command-line use
        recommender = ProfessionalCropRecommender()
        
        if not recommender.load_and_merge_data():
            if __name__ == "__main__":
                print("Failed to load crop data")
            return None
            
        model_info = recommender.train_model()
        if not model_info:
            if __name__ == "__main__":
                print("Model training failed")
            return None
            
        # Get inputs from command line arguments or stdin
        try:
            user_input = {
                'soil_ph': float(os.getenv('SOIL_PH') or input("Soil pH (3.5-9.5): ")),
                'soil_temp': float(os.getenv('SOIL_TEMP') or input("Soil temperature (°C): ")),
                'soil_type': os.getenv('SOIL_TYPE') or input(f"Soil type ({', '.join(recommender.VALID_SOIL_TYPES)}): ").strip().capitalize(),
                'rainfall': int(os.getenv('RAINFALL') or input("Annual rainfall (mm): ")),
                'humidity': float(os.getenv('HUMIDITY') or input("Average humidity (%): "))
            }
            
            recommender.validate_inputs(**user_input)
            suggestions = recommender.generate_suggestions(user_input)
            
            if suggestions:
                if __name__ == "__main__":
                    print(f"Recommended Crop: {suggestions[0]['crop']}")
                return suggestions[0]['crop']
            return None
        except Exception as e:
            if __name__ == "__main__":
                print(f"Error: {str(e)}")
            return None
    else:
        def main():
            """Enhanced main function with interactive menu"""
    print("\n🌱 PROFESSIONAL CROP RECOMMENDER v6.1 🌱")
    print("="*50)
    print("Advanced crop recommendations with weather, market intelligence and AI chat (Gemini)\n")
    
    try:
        recommender = ProfessionalCropRecommender()
        
        # Load data
        print("\n🔄 Loading crop data...")
        data_files = [
            ('basic', 'crop.csv'),
            ('professional', 'professional_crops.csv')
        ]
        for name, path in data_files:
            if os.path.exists(path):
                print(f" - Found {name} data file")
            else:
                print(f"⚠️ Missing {name} data file: {path}")
        
        if not recommender.load_and_merge_data():
            raise RuntimeError("Failed to load crop data")
        
        # Train model
        print("\n🤖 Training recommendation model (this may take a few minutes)...")
        model_info = recommender.train_model()
        if not model_info:
            raise RuntimeError("Model training failed")
        print(f"✅ Model trained with accuracy: {model_info['best_score']:.1%}")
        if model_info['best_score'] < 0.5:
            print("⚠️ Warning: Model accuracy is below 50%. Consider collecting more data.")
        
        while True:
            print("\nOptions:")
            print("1. Get crop recommendations")
            print("2. Ask farming question (Gemini AI)")
            print("3. Exit")
            
            choice = input("\nChoose an option (1-3): ").strip()
            
            if choice == '1':
                # Get crop recommendations
                print("\n📝 Enter your growing conditions:")
                user_input = {
                    'soil_ph': float(input(" - Soil pH (3.5-9.5): ")),
                    'soil_temp': float(input(" - Soil temperature (°C): ")),
                    'soil_type': input(f" - Soil type ({', '.join(recommender.VALID_SOIL_TYPES)}): ").strip().capitalize(),
                    'rainfall': int(input(" - Annual rainfall (mm): ")),
                    'humidity': float(input(" - Average humidity (%): "))
                }
                location = input(" - Location (optional, for weather analysis): ").strip() or None
                
                recommender.validate_inputs(**user_input)
                print("\n🔍 Analyzing best crops for your conditions and weather...")
                suggestions = recommender.generate_suggestions(user_input, location)
                
                if not suggestions:
                    print("\n⚠️ No crops matched your exact conditions. Showing closest matches:")
                    recommender.debug_conditions(user_input)
                    continue
                
                report = recommender.generate_report(suggestions, model_info, user_input, location)
                print(f"\n📊 Recommendation report generated:\n{report}")
                
                # Save reports
                txt_file = recommender.save_report_to_file(report)
                html_file = recommender.save_report_as_html(report)
                
                if txt_file:
                    print(f"\n📝 Text report saved to: {txt_file}")
                if html_file:
                    print(f"🌐 Full report with visuals saved to: {html_file}")
                
                plot_path = recommender.plot_recommendations(suggestions)
                if plot_path:
                    print(f"📈 Visualization saved to: {plot_path}")
                    
            elif choice == '2':
                # Chat with AI assistant
                print("\n💬 Ask me anything about crops, fertilizers, or farming practices")
                print("Examples:")
                print("- What's the current price trend for wheat?")
                print("- How should I fertilize my tomato plants?")
                print("- What are common diseases affecting rice crops?")
                print("- Type 'back' or 'exit' to return to main menu")
                
                question = input("\nYour question: ").strip()
                if question.lower() in ('exit', 'quit', 'back'):
                    continue
                    
                print("\n🤖 Thinking...")
                response = recommender.chat_with_assistant(question)
                print("\n" + "="*60)
                print(response)
                print("="*60)
                
                # Option to save response
                if input("\nSave this response to a file? (y/n): ").lower() == 'y':
                    filename = f"agriculture_advice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Question: {question}\n\nAnswer:\n{response}")
                    print(f"✅ Response saved to {filename}")
                    
            elif choice == '3':
                print("\nThank you for using the Professional Crop Recommender!")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print(" - Check that all data files exist")
        print(" - Verify your input values are within valid ranges")
        print(" - For API errors, check your internet connection")
    finally:
        print("\n🙏 Thank you for using the Professional Crop Recommender!")

if __name__ == "__main__":
    main()