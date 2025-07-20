# --- START OF FILE me.py ---

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta, timezone # Ensure timezone is imported
import yfinance as yf
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import requests # Needed for weather API
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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Key Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Explicitly check for Gemini key
if not GEMINI_API_KEY:
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Please set it in a .env file in the script's directory.")

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_recommender.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Warnings and Options ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# --- Enums ---
class WeatherRiskLevel(Enum):
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    EXTREME = auto()

    def __str__(self):
        return self.name

# --- AI Assistant Class ---
class AgricultureChatAssistant:
    """AI assistant for answering agriculture-related questions using Google Gemini API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key not configured")
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            logger.error(f"Failed to configure Google Gemini: {e}")
            raise ValueError(f"Failed to configure Google Gemini: {e}")

        self.base_prompt = """You are an expert agricultural assistant specialized in crop cultivation,
        market prices, fertilizers, and farming techniques. Provide accurate, practical advice
        tailored to small and large-scale farmers. Be concise but thorough in your explanations.

        When asked about:
        - Crop prices: Provide current market trends and factors affecting prices. If available, mention checking CROP_MARKET_DATA for reference values.
        - Fertilizers: Recommend appropriate types and application methods.
        - Cultivation: Offer best practices for soil preparation, planting, and care, including details like planting depth, spacing, watering needs, and harvesting indicators.
        - Pests/Diseases: List common pests and diseases for a crop, describe their damage, and suggest both organic and chemical control methods, including preventative measures.
        - Any other agriculture topic: Provide expert guidance.

        Always remind farmers to consult local agricultural extension services for specific regional advice.
        """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def ask_question(self, question: str, max_tokens: int = 1000) -> str:
        """Ask an agriculture-related question to the AI assistant"""
        try:
            if not hasattr(self, 'model'):
                raise RuntimeError("Gemini model not initialized.")

            logger.info(f"Sending question to Gemini (first 50 chars): {question[:50]}...")
            response = self.model.generate_content(
                f"{self.base_prompt}\n\nQuestion: {question}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            if hasattr(response, 'text') and response.text:
                return response.text
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return "".join(part.text for part in response.candidates[0].content.parts)
            else:
                logger.error(f"Unexpected Gemini API response structure: {response}")
                return "I'm sorry, I received an unexpected response structure from the AI."

        except Exception as e:
            logger.error(f"Gemini API error during generation: {str(e)}", exc_info=True)
            if "API key not valid" in str(e):
                return "I'm sorry, the configured API key is not valid."
            elif "quota" in str(e).lower():
                return "I'm sorry, the request quota has been exceeded."
            return f"I'm sorry, I couldn't process your question due to an API error: {str(e)}."

# --- Main Recommender Class ---
class ProfessionalCropRecommender:
    """Advanced crop recommendation system with weather, market intelligence, and AI chat"""

    # Class constants
    NUMERIC_FEATURES = ['Soil pH Min', 'Soil pH Max',
                        'Soil Temp Min (°C)', 'Soil Temp Max (°C)',
                        'Rainfall Min (mm/year)', 'Rainfall Max (mm/year)',
                        'Humidity Min (%)', 'Humidity Max (%)']
    CATEGORICAL_FEATURES = ['Soil Type']
    TARGET_COLUMN = 'Crop'
    VALID_SOIL_TYPES = ['Loam', 'Clay', 'Sandy', 'Siltyclay', 'Clayloam', 'Sandyloam', 'Peat', 'Siltloam']
    CACHE_DIR = "cache"
    PRICES_FILE = "crop_prices.json"
    SEASONAL_CACHE_DAYS = 1
    REQUEST_DELAY = 1.1
    MIN_CONFIDENCE_THRESHOLD = 0.05
    MODEL_FILE = "pretrained_model.joblib"
    PREPROCESSOR_FILE = "pretrained_preprocessor.joblib"
    DATA_CACHE_FILE = "crop_data_cache.joblib"
    CROP_MARKET_DATA = {
        'Wheat': {'ticker': 'KE=F', 'profit_per_acre': 400, 'unit': 'bushel', 'fallback_price': 6.50},
        'Corn': {'ticker': 'ZC=F', 'profit_per_acre': 600, 'unit': 'bushel', 'fallback_price': 4.20},
        'Soybeans': {'ticker': 'ZS=F', 'profit_per_acre': 500, 'unit': 'bushel', 'fallback_price': 12.00},
        'Rice': {'ticker': 'ZR=F', 'profit_per_acre': 450, 'unit': 'hundredweight', 'fallback_price': 15.50},
        'Cotton': {'ticker': 'CT=F', 'profit_per_acre': 550, 'unit': 'pound', 'fallback_price': 0.85},
        'Potatoes': {'profit_per_acre': 800, 'unit': 'ton', 'fallback_price': 150.00},
        'Tomatoes': {'profit_per_acre': 1200, 'unit': 'ton', 'fallback_price': 80.00},
        'Apples': {'profit_per_acre': 1000, 'unit': 'bushel', 'fallback_price': 25.00},
        'Oranges': {'profit_per_acre': 950, 'unit': 'box', 'fallback_price': 12.00},
        'Blueberries': {'profit_per_acre': 1500, 'unit': 'pound', 'fallback_price': 3.50}
    }

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.df: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.geolocator = Nominatim(user_agent="crop_recommender_pro_v9_meteo", timeout=15)
        self.price_db = {}
        self.weather_cache = {}
        try:
            self.chat_assistant = AgricultureChatAssistant(GEMINI_API_KEY)
        except ValueError as e:
            logger.error(f"Failed to initialize Chat Assistant: {e}")
            self.chat_assistant = None

        self._setup_cache()
        self._load_price_database()
        self.load_model()
        self.load_and_merge_data()

    # --- Cache Management ---
    def _setup_cache(self):
        """Initialize cache directory."""
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed setup cache dir '{self.CACHE_DIR}': {e}")

    def _load_cache(self, cache_type: str, key: str) -> Optional[dict]:
        """Load cached data if available and recent."""
        safe_key = "".join(c for c in key if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_key:
            logger.warning(f"Invalid cache key for load: {key}")
            return None
        cache_file = os.path.join(self.CACHE_DIR, f"{cache_type}_{safe_key}.json")
        if not os.path.exists(cache_file):
            return None
        try:
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mod_time) > timedelta(days=self.SEASONAL_CACHE_DAYS):
                logger.info(f"Cache expired: {cache_file}")
                return None
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'last_updated' not in data:
                logger.warning(f"{cache_file} missing 'last_updated'. Invalid.")
                return None
            logger.info(f"Using cached data: {cache_file}")
            return data
        except (json.JSONDecodeError, IOError, UnicodeDecodeError, ValueError, OSError) as e:
            logger.warning(f"Cache load error {cache_file}: {e}. Removing.")
            try:
                os.remove(cache_file)
            except OSError as rm_e:
                logger.error(f"Failed remove corrupted cache {cache_file}: {rm_e}")
            return None

    def _save_cache(self, data: dict, cache_type: str, key: str):
        """Save data to cache with UTC timestamp."""
        safe_key = "".join(c for c in key if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_key:
            logger.warning(f"Invalid cache key for save: {key}")
            return
        cache_file = os.path.join(self.CACHE_DIR, f"{cache_type}_{safe_key}.json")
        try:
            data['last_updated'] = datetime.now(timezone.utc).isoformat()
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except (IOError, TypeError) as e:
            logger.warning(f"Failed save cache {cache_file}: {e}")

    # --- Price Database ---
    def _load_price_database(self):
        """Load price database from file."""
        if not os.path.exists(self.PRICES_FILE):
            logger.info(f"'{self.PRICES_FILE}' not found.")
            self.price_db = {}
            return
        try:
            with open(self.PRICES_FILE, 'r', encoding='utf-8') as f:
                loaded_db = json.load(f)
            self.price_db = {k: v for k, v in loaded_db.items() if isinstance(v, dict)}
            if len(self.price_db) != len(loaded_db):
                logger.warning("Removed invalid entries from price DB.")
        except (IOError, json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed load price DB '{self.PRICES_FILE}': {e}. Starting fresh.")
            self.price_db = {}

    def _save_price_database(self):
        """Save price database to file."""
        try:
            with open(self.PRICES_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.price_db, f, indent=2, default=str)
        except IOError as e:
            logger.warning(f"Failed save price DB '{self.PRICES_FILE}': {e}")

    # --- Model Persistence ---
    def save_model(self):
        """Save the trained model and preprocessor."""
        try:
            os.makedirs(os.path.dirname(self.MODEL_FILE) or '.', exist_ok=True)
            if self.model and self.preprocessor:
                joblib.dump(self.model, self.MODEL_FILE)
                joblib.dump(self.preprocessor, self.PREPROCESSOR_FILE)
                logger.info(f"Model saved to {self.MODEL_FILE}, Preprocessor to {self.PREPROCESSOR_FILE}")
                return True
            logger.warning("No model/preprocessor to save.")
            return False
        except Exception as e:
            logger.error(f"Failed save model/preprocessor: {e}")
            return False

    def load_model(self) -> bool:
        """Load pre-trained model and preprocessor."""
        m_exists, p_exists = os.path.exists(self.MODEL_FILE), os.path.exists(self.PREPROCESSOR_FILE)
        if m_exists and p_exists:
            try:
                self.model, self.preprocessor = joblib.load(self.MODEL_FILE), joblib.load(self.PREPROCESSOR_FILE)
                if not isinstance(self.model, Pipeline) or not isinstance(self.preprocessor, ColumnTransformer):
                    logger.error("Loaded files aren't Pipeline/ColumnTransformer.")
                    self.model, self.preprocessor = None, None
                    return False
                logger.info("Pre-trained model/preprocessor loaded.")
                return True
            except Exception as e:
                logger.error(f"Failed load model/preprocessor: {e}")
                self.model, self.preprocessor = None, None
                return False
        else:
            logger.info(f"Model/preprocessor files missing. Need training.")
            return False

    # --- AI Interaction ---
    def chat_with_assistant(self, question: str) -> str:
        """Interact with the agriculture expert assistant."""
        if not self.chat_assistant:
            return "AI Chat Assistant unavailable."
        if not question.strip():
            return "Please ask a question."
        return self.chat_assistant.ask_question(question)

    # --- Report Generation ---
    def generate_report(self, suggestions: List[dict], model_info: dict,
                       user_input: dict, location: Optional[str] = None,
                       weather_data: Optional[dict] = None) -> str:
        """Generate detailed report with comparisons and weather."""
        if not suggestions:
            return "No crop suggestions available."

        report = [f"🌱 CROP RECOMMENDATION REPORT for {location or 'specified conditions'} 🌱"]
        report.append("=" * len(report[0]))
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # User Input
        report.append("\n📝 Your Conditions:")
        report.append(f"  - Soil pH: {user_input['soil_ph']}")
        report.append(f"  - Soil Temp: {user_input['soil_temp']} °C")
        report.append(f"  - Soil Type: {user_input['soil_type'].capitalize()}")
        report.append(f"  - Rainfall: {user_input['rainfall']} mm/year")
        report.append(f"  - Humidity: {user_input['humidity']} %")

        # Current Weather
        report.append("\n🌦️ Current Weather Conditions" + (f" ({location})" if location else ""))
        if weather_data:
            report.append(f"  - Location: {weather_data.get('city_name', location or 'N/A')}")
            report.append(f"  - Conditions: {weather_data.get('description', 'N/A').capitalize()}")
            temp = weather_data.get('temperature', 'N/A')
            feels = weather_data.get('feels_like', 'N/A')
            report.append(f"  - Temperature: {temp}°C (Feels like: {feels}°C)")
            report.append(f"  - Humidity: {weather_data.get('humidity', 'N/A')}%")
            report.append(f"  - Wind: {weather_data.get('wind_speed', 'N/A')} km/h")
            precip = weather_data.get('precipitation', 'N/A')
            report.append(f"  - Precipitation (current): {precip} mm")
            fetch_time_str = weather_data.get('fetch_time_utc')
            if fetch_time_str:
                try:
                    report.append(f"  - Fetched at: {datetime.fromisoformat(fetch_time_str).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                except ValueError:
                    report.append(f"  - Fetched at: {fetch_time_str} (unparsed)")
            else:
                report.append("  - Fetched at: N/A")
        elif location:
            report.append("  - Weather data could not be fetched.")
        else:
            report.append("  - Location not provided.")

        # Recommendations & Comparison
        report.append("\n🌟 Top Recommendations & Comparison:")
        for i, sug in enumerate(suggestions):
            crop_name = sug['Crop']
            report.append(f"\n{i+1}. {crop_name} (Confidence: {sug['Confidence']:.1%})")
            crop_req = self.df[self.df[self.TARGET_COLUMN] == crop_name].iloc[0] if self.df is not None and not self.df[self.df[self.TARGET_COLUMN] == crop_name].empty else None
            report.append("   ------------------------------------------------------")
            report.append("   | Condition         | Your Input      | Crop Needs               |")
            report.append("   |-------------------|-----------------|--------------------------|")
            if crop_req is not None:
                def format_range(min_col, max_col, unit=""):
                    min_v, max_v = crop_req.get(min_col), crop_req.get(max_col)
                    if pd.notna(min_v) and pd.notna(max_v): return f"{min_v:.1f} - {max_v:.1f}{unit}"
                    if pd.notna(min_v): return f">= {min_v:.1f}{unit}"
                    if pd.notna(max_v): return f"<= {max_v:.1f}{unit}"
                    return "N/A"
                ph_needs = format_range('Soil pH Min', 'Soil pH Max')
                temp_needs = format_range('Soil Temp Min (°C)', 'Soil Temp Max (°C)', ' °C')
                rain_needs = format_range('Rainfall Min (mm/year)', 'Rainfall Max (mm/year)', ' mm')
                hum_needs = format_range('Humidity Min (%)', 'Humidity Max (%)', ' %')
                soil_needs = crop_req.get('Soil Type', 'N/A')
            else:
                ph_needs, temp_needs, rain_needs, hum_needs, soil_needs = ('N/A',) * 5
                if self.df is not None:
                    report.append("   | Error: Requirements not found for this crop. |                   |")
            report.append(f"   | Soil pH           | {user_input['soil_ph']:<15.1f} | {ph_needs:<24} |")
            report.append(f"   | Soil Temp (°C)    | {user_input['soil_temp']:<15.1f} | {temp_needs:<24} |")
            report.append(f"   | Rainfall (mm/yr)  | {user_input['rainfall']:<15} | {rain_needs:<24} |")
            report.append(f"   | Humidity (%)      | {user_input['humidity']:<15.1f} | {hum_needs:<24} |")
            report.append(f"   | Soil Type         | {user_input['soil_type'].capitalize():<15} | {soil_needs:<24} |")
            report.append("   ------------------------------------------------------")
            # Profitability & Risk
            profit_str = "N/A"
            if 'EstimatedProfitability' in sug:
                price = sug.get('MarketPrice', 'N/A')
                unit = sug.get('PriceUnit', 'unit')
                source = sug.get('PriceSource', 'N/A')
                profit = sug.get('EstimatedProfitability', 0)
                price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
                profit_str = f"${profit:.2f}/acre (Price: {price_str}/{unit}, Source: {source})"
            report.append(f"   - Est. Profit: {profit_str}")
            if 'WeatherRisk' in sug:
                report.append(f"   - General Weather Risk: {sug['WeatherRisk']}")

        # Footer Notes
        report.append("\n💡 Notes:")
        report.append(" - Crop Needs show typical dataset ranges.")
        report.append(" - Confidence = model's certainty.")
        report.append(" - Profitability is an estimate.")
        report.append(" - Weather risk based on general assessment.")
        report.append(" - Consult local experts for specific advice.")
        return "\n".join(report)

    def save_report_to_file(self, report_text: str, filename: str = "crop_recommendations.txt"):
        """Save the recommendation report to a text file."""
        filepath = os.path.join(os.getcwd(), filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Text report saved to {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Failed save text report {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving text report: {e}")
            return None

    def save_report_as_html(self, report_text: str, filename: str = "crop_recommendations.html"):
        """Save report as HTML with basic styling."""
        filepath = os.path.join(os.getcwd(), filename)
        try:
            html_content = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Crop Recommendations</title><style>body{{font-family:'Courier New',monospace;line-height:1.4;padding:20px;background-color:#f9f9f9;}}h1{{color:#2E7D32;border-bottom:2px solid #2E7D32;padding-bottom:5px;}}pre{{background-color:#fff;padding:15px;border:1px solid #e0e0e0;border-radius:5px;white-space:pre-wrap;word-wrap:break-word;font-size:0.9em;box-shadow:2px 2px 5px rgba(0,0,0,0.1);}}em{{color:#555;font-size:0.8em;}}</style></head><body><h1>Crop Recommendation Report</h1><pre>{report_text}</pre><p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p></body></html>"""
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTML report saved to {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"Failed save HTML report {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving HTML report: {e}")
            return None

    # --- Plotting ---
    def plot_recommendations(self, suggestions: List[dict]) -> Optional[str]:
        """Create bar chart visualization of recommendation confidence."""
        if not suggestions:
            logger.warning("No suggestions for plotting.")
            return None
        try:
            crops = [s['Crop'] for s in suggestions]
            confidences = [s['Confidence'] * 100 for s in suggestions]
            plt.figure(figsize=(10, max(6, len(crops) * 0.6)))
            barplot = sns.barplot(x=confidences, y=crops, palette="viridis", hue=crops, dodge=False, legend=False)
            plt.xlabel("Model Confidence (%)")
            plt.ylabel("Crop")
            plt.title("Top Crop Recommendations by Confidence")
            plt.xlim(0, 105)
            for i, bar in enumerate(barplot.patches):
                plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{confidences[i]:.1f}%', va='center', ha='left')
            plt.tight_layout()
            plot_filename = f"crop_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_filepath = os.path.join(os.getcwd(), plot_filename)
            plt.savefig(plot_filepath)
            plt.close()
            logger.info(f"Plot saved: {plot_filepath}")
            return plot_filepath
        except ImportError:
            logger.warning("Matplotlib/seaborn not found. Plotting skipped.")
            return None
        except Exception as e:
            logger.error(f"Failed generate plot: {e}")
            return None

    # --- Debugging ---
    def debug_conditions(self, user_input: dict):
        """Show debugging info for user conditions."""
        print("\n--- Debug User Input ---")
        for k, v in user_input.items():
            print(f" - {k}: {v} ({type(v).__name__})")
        ph_min, ph_max = 3.0, 10.0
        if not (ph_min <= user_input['soil_ph'] <= ph_max):
            print(f" - WARN: Soil pH outside {ph_min}-{ph_max}")
        print("--- End Debug ---")

    # --- Data Loading ---
    def load_and_merge_data(self) -> bool:
        """Load/merge crop data from CSV, using cache if recent."""
        cache_valid = False
        if os.path.exists(self.DATA_CACHE_FILE):
            try:
                cache_time = datetime.fromtimestamp(os.path.getmtime(self.DATA_CACHE_FILE))
                if (datetime.now() - cache_time) < timedelta(days=1):
                    self.df = joblib.load(self.DATA_CACHE_FILE)
                    if isinstance(self.df, pd.DataFrame) and not self.df.empty and self.TARGET_COLUMN in self.df.columns:
                        logger.info(f"Loaded valid cached crop data ({len(self.df)} rows)")
                        cache_valid = True
                    else:
                        logger.warning("Data cache invalid/empty. Reloading.")
                else:
                    logger.info("Data cache expired. Reloading.")
            except Exception as e:
                logger.error(f"Failed load/validate cached data: {e}. Reloading.")
        if cache_valid:
            return True

        logger.info("Loading data from CSV...")
        try:
            basic_csv = 'crop.csv'
            if not os.path.exists(basic_csv):
                logger.error(f"CRITICAL: '{basic_csv}' not found.")
                self.df = None
                return False
            basic_df = pd.read_csv(basic_csv)
            pro_csv = 'professional_crops.csv'
            if os.path.exists(pro_csv):
                try:
                    pro_df = pd.read_csv(pro_csv)
                    if set(basic_df.columns) == set(pro_df.columns):
                        self.df = pd.concat([basic_df, pro_df], ignore_index=True)
                        logger.info("Merged datasets.")
                    else:
                        logger.warning(f"Column mismatch {basic_csv}/{pro_csv}. Using basic.")
                        self.df = basic_df
                except Exception as e:
                    logger.warning(f"Failed load/merge {pro_csv}: {e}. Using basic.")
                    self.df = basic_df
            else:
                self.df = basic_df
            # Cleaning
            if self.TARGET_COLUMN not in self.df.columns:
                logger.error(f"Target '{self.TARGET_COLUMN}' missing.")
                self.df = None
                return False
            self.df = self.df.dropna(subset=[self.TARGET_COLUMN])
            required = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
            if not all(col in self.df.columns for col in required):
                logger.error(f"Missing features: {[c for c in required if c not in self.df.columns]}")
                self.df = None
                return False
            for col in self.NUMERIC_FEATURES:
                if self.df[col].isnull().any(): # Check before filling
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            for col in self.CATEGORICAL_FEATURES:
                 if self.df[col].isnull().any(): # Check before filling
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

            min_samples=3
            counts=self.df[self.TARGET_COLUMN].value_counts()
            valid=counts[counts >= min_samples].index
            self.df=self.df[self.df[self.TARGET_COLUMN].isin(valid)].copy()
            if len(valid) < len(counts):
                logger.warning(f"Removed {len(counts)-len(valid)} crops <{min_samples} samples.")
            if self.df.empty:
                logger.error("No valid data after filtering.")
                return False
            # Save Cache
            try:
                joblib.dump(self.df, self.DATA_CACHE_FILE)
                logger.info(f"Processed data saved to cache.")
            except Exception as e:
                logger.error(f"Failed save data cache: {e}")
            logger.info(f"Loaded/processed data: {len(self.df)} rows, {len(valid)} crops.")
            return True
        except (pd.errors.EmptyDataError, FileNotFoundError) as ferr:
            logger.error(f"Error reading CSV: {ferr}")
            self.df = None
            return False
        except Exception as e:
            logger.error(f"Failed load/process data: {e}", exc_info=True)
            self.df = None
            return False

    # --- Input Validation ---
    def validate_inputs(self, soil_ph: float, soil_temp: float, soil_type: str, rainfall: int, humidity: float):
        """Validate user inputs against plausible ranges/types."""
        errors = []
        if not (3.0 <= soil_ph <= 10.0):
            errors.append(f"Soil pH ({soil_ph}) out of range [3.0, 10.0].")
        if soil_temp < -20 or soil_temp > 60:
            errors.append(f"Soil temp ({soil_temp}°C) outside [-20, 60].")
        norm_st = soil_type.strip().replace(" ", "").lower()
        valid_st = [s.lower() for s in self.VALID_SOIL_TYPES]
        if norm_st not in valid_st:
            errors.append(f"Invalid soil type '{soil_type}'. Valid: {', '.join(self.VALID_SOIL_TYPES)}")
        if rainfall < 0 or rainfall > 12000:
            errors.append(f"Rainfall ({rainfall}mm) outside [0, 12000].")
        if not (0 <= humidity <= 100):
            errors.append(f"Humidity ({humidity}%) must be 0-100.")
        if errors:
            raise ValueError("Input validation failed:\n" + "\n".join(f"- {e}" for e in errors))
        logger.info("User inputs validated.")

    # --- Model Training ---
    def train_model(self) -> Optional[dict]:
        """Train recommendation model with hyperparameter tuning."""
        if self.df is None or self.df.empty:
            logger.error("No data for training.")
            return None
        if self.TARGET_COLUMN not in self.df.columns:
            logger.error(f"Target '{self.TARGET_COLUMN}' missing.")
            return None

        logger.info("Starting model training...")
        try:
            X = self.df[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]
            y = self.df[self.TARGET_COLUMN]
            if X.empty or y.empty:
                logger.error("X or y empty.")
                return None

            counts=y.value_counts()
            min_s=counts.min()
            n_s=min(5, min_s)
            if n_s < 2:
                raise ValueError(f"Smallest class ('{counts.idxmin()}') has {min_s} samples. Need >= 2 for CV.")

            logger.info(f"Using {n_s}-fold Stratified CV (min class count: {min_s})")
            num_t=Pipeline([('imp',SimpleImputer(strategy='median'))])
            cat_t=Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                            ('ohe',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            self.preprocessor=ColumnTransformer([('num', num_t, self.NUMERIC_FEATURES),
                                                 ('cat', cat_t, self.CATEGORICAL_FEATURES)],
                                                remainder='passthrough')
            pipe=Pipeline([('prep', self.preprocessor),
                           ('clf', RandomForestClassifier(random_state=42))]) # Placeholder classifier

            p_grid=[
                {'clf': [RandomForestClassifier(random_state=42)],
                 'clf__n_estimators': [100, 150],
                 'clf__max_depth': [None, 15],
                 'clf__min_samples_split': [2, 5],
                 'clf__class_weight': ['balanced']},
                {'clf': [GradientBoostingClassifier(random_state=42)],
                 'clf__n_estimators': [100, 150],
                 'clf__learning_rate': [0.1],
                 'clf__max_depth': [3, 5]}
            ]
            cv_s=StratifiedKFold(n_splits=n_s, shuffle=True, random_state=42)
            gs=GridSearchCV(pipe, p_grid, cv=cv_s, scoring='accuracy', n_jobs=-1, verbose=1)

            logger.info("Fitting GridSearchCV...")
            gs.fit(X, y)

            self.model=gs.best_estimator_
            self.preprocessor=self.model.named_steps['prep'] # Get fitted preprocessor from best pipeline

            logger.info(f"Training complete. Best Score: {gs.best_score_:.4f}, Best Params: {gs.best_params_}")
            self.save_model() # Save the newly trained model

            return {'best_score':gs.best_score_, 'best_params':gs.best_params_,
                    'n_splits_used':n_s, 'min_class_count':min_s,
                    'classifier_type':type(self.model.named_steps['clf']).__name__}

        except ValueError as ve:
            logger.error(f"Config error training: {ve}")
            return None
        except MemoryError:
            logger.error("Memory Error training.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error training: {e}", exc_info=True)
            self.model, self.preprocessor = None, None
            return None

    # --- Geocoding and Weather ---
    @lru_cache(maxsize=20)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6),
           retry_error_callback=lambda rs: logger.warning(f"Geocoder failed for '{rs.args[1]}' after {rs.attempt_number} attempts."))
    def _get_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get latitude and longitude using geopy."""
        if not location_name:
            return None
        try:
            logger.info(f"Geocoding: {location_name}")
            time.sleep(self.REQUEST_DELAY)
            loc = self.geolocator.geocode(location_name)
            if loc:
                logger.info(f"Coords: ({loc.latitude:.4f}, {loc.longitude:.4f})")
                return (loc.latitude, loc.longitude)
            logger.warning(f"Could not geocode: {location_name}")
            return None
        except (GeocoderTimedOut, GeocoderServiceError) as geo_err:
            logger.error(f"Geocoder error for {location_name}: {geo_err}")
            return None
        except Exception as e:
            logger.error(f"Unexpected geocoding error for {location_name}: {e}")
            return None

    # --- Open-Meteo Weather Fetching ---
    def _get_weather_description(self, code: int) -> str:
        """Convert WMO weather code to human-readable description."""
        wmo_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            77: "Snow grains",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }
        return wmo_codes.get(code, f"Unknown code ({code})")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6),
           retry_error_callback=lambda rs: logger.warning(f"Open-Meteo API failed for ({rs.args[1]}, {rs.args[2]}) after {rs.attempt_number} attempts."))
    def _fetch_weather_api(self, lat: float, lon: float) -> Optional[dict]:
        """Fetch current weather data from Open-Meteo API."""
        api_endpoint = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat, 'longitude': lon,
            'current': 'temperature_2m,relativehumidity_2m,apparent_temperature,precipitation,weathercode,windspeed_10m',
            'timezone': 'auto'
        }
        try:
            logger.info(f"Fetching current weather from Open-Meteo for ({lat:.4f}, {lon:.4f})")
            response = requests.get(api_endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info("Successfully fetched weather data from Open-Meteo.")

            current = data.get('current')
            if not current:
                logger.warning("Open-Meteo response missing 'current' data.")
                return None

            weather_code = current.get('weathercode')
            description = self._get_weather_description(weather_code) if weather_code is not None else "N/A"

            current_weather = {
                "temperature": current.get("temperature_2m"),
                "feels_like": current.get("apparent_temperature"),
                "humidity": current.get("relativehumidity_2m"),
                "precipitation": current.get("precipitation"),
                "description": description,
                "weathercode": weather_code,
                "wind_speed": current.get("windspeed_10m"),
                "city_name": "N/A", # Not provided by this API endpoint
                "country": "N/A", # Not provided by this API endpoint
                "fetch_time_utc": datetime.now(timezone.utc).isoformat(),
                # Placeholder forecast data (CRUDE)
                "temp_avg_next_7d": current.get("temperature_2m", 20),
                "precip_chance_next_7d": 0.5 if current.get("precipitation", 0) > 0 else 0.1,
                "extreme_weather_alert": weather_code in [95, 96, 99] if weather_code is not None else False
            }
            return current_weather

        except requests.exceptions.Timeout:
            logger.error("Weather API request timed out.")
            return None
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching weather: {http_err} - {response.text}")
            return None
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Weather API request failed: {req_err}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Failed decode JSON from Weather API: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}", exc_info=True)
            return None

    def get_weather_data(self, location: str) -> Optional[dict]:
        """Get weather data, using cache if available and recent."""
        coords = self._get_coordinates(location)
        if not coords:
            return None
        lat, lon = coords
        cache_key = f"{lat:.4f}_{lon:.4f}"
        cached_data = self._load_cache("weather", cache_key)
        if cached_data:
            return cached_data.get('weather_info')

        logger.info(f"Fetching fresh weather data for {location}")
        weather_info = self._fetch_weather_api(lat, lon)
        if weather_info:
            self._save_cache({"weather_info": weather_info}, "weather", cache_key)
            return weather_info
        logger.warning(f"Could not fetch weather data for {location}")
        return None

    def _analyze_weather_risk(self, weather_data: Optional[dict]) -> WeatherRiskLevel:
        """Assess general weather risk based on fetched/dummy forecast data."""
        if not weather_data:
            return WeatherRiskLevel.MODERATE
        risk = WeatherRiskLevel.LOW
        try:
            if weather_data.get("extreme_weather_alert", False):
                return WeatherRiskLevel.EXTREME
            temp = float(weather_data.get("temp_avg_next_7d", 20))
            precip_chance = float(weather_data.get("precip_chance_next_7d", 0.3))
            if temp > 38 or temp < 2:
                risk = WeatherRiskLevel.HIGH
            elif temp > 32 or temp < 8:
                risk = max(risk, WeatherRiskLevel.MODERATE)
            if precip_chance > 0.8 or precip_chance < 0.05:
                risk = max(risk, WeatherRiskLevel.MODERATE)
        except (TypeError, ValueError):
            logger.warning("Could not parse weather data for risk analysis.")
            return WeatherRiskLevel.MODERATE
        return risk

    # --- Market Data ---
    @lru_cache(maxsize=30)
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4),
           retry_error_callback=lambda rs: logger.warning(f"yfinance failed for {rs.args[1]} after {rs.attempt_number} attempts"))
    def _fetch_stock_price(self, ticker: str) -> Optional[float]:
        """Fetch stock price using yfinance."""
        try:
            logger.info(f"Fetching market price: {ticker}")
            stock=yf.Ticker(ticker)
            info=stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('bid') or info.get('dayHigh') or info.get('previousClose')
            if price:
                logger.info(f"Price found {ticker}: {price}")
                return float(price)
            logger.warning(f"No price field {ticker}. Fields: {list(info.keys())[:10]}...")
            return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as req_err:
            logger.error(f"Network error price {ticker}: {req_err}")
            return None
        except IndexError:
            logger.warning(f"Data structure error price {ticker}")
            return None
        except Exception as e:
            logger.error(f"Error fetching price {ticker}: {e}")
            return None

    def get_market_data(self, crop_name: str) -> dict:
        """Get market price/profitability via cache/DB/API/fallback."""
        market_info = self.CROP_MARKET_DATA.get(crop_name, {})
        ticker=market_info.get('ticker')
        fallback=market_info.get('fallback_price')
        profit=market_info.get('profit_per_acre',0)
        unit=market_info.get('unit','unit')
        price=None
        source="start"
        cache_key=f"price_{crop_name.replace(' ','_')}"
        cached=self._load_cache("price", cache_key)

        if cached:
            db_entry=cached.get('price_info')
            price=db_entry.get('price') if isinstance(db_entry, dict) else None
            source="cache" if price is not None else source

        if price is None:
            db_entry=self.price_db.get(crop_name)
            price=db_entry.get('price') if isinstance(db_entry, dict) else None
            source="database" if price is not None else source

        if price is None and ticker:
            fetched=self._fetch_stock_price(ticker)
            price=fetched if fetched is not None else price
            source="live API" if fetched is not None else "API fetch failed"

        if price is None:
            price=fallback
            source="fallback" if fallback is not None else "unavailable"

        if source in ["live API", "database"] and price is not None and not cached:
            price_info={'price':price,'unit':unit,'timestamp':datetime.now(timezone.utc).isoformat()}
            self.price_db[crop_name]=price_info
            self._save_price_database()
            self._save_cache({'price_info':price_info},"price",cache_key)

        logger.info(f"Price source for {crop_name}: {source}, Price: {price}")
        est_profit=profit
        if isinstance(price, (int, float)) and isinstance(fallback, (int, float)) and fallback > 0:
             try: # Add try-except for safety
                 est_profit=profit*(price/fallback)
             except ZeroDivisionError: # Should not happen with check > 0, but good practice
                 logger.warning(f"Fallback price is zero for {crop_name}, cannot calculate ratio.")
        elif price is None:
            est_profit=0 # Indicate uncertainty

        return {"Price":price, "Unit":unit, "PriceSource":source,
                "EstimatedProfitability":est_profit, "MarketPrice":price}


    # --- Core Suggestion Generation ---
    def generate_suggestions(self, user_input: dict, location: Optional[str] = None) -> Tuple[List[dict], Optional[dict]]:
        """Generate crop suggestions, fetch weather. Returns (suggestions, weather_data)."""
        if not self.model or not self.preprocessor:
            logger.error("Model/Preprocessor missing.")
            return [], None

        logger.info("Generating suggestions...")
        suggestions=[]
        weather_data=None
        try:
            input_data={
                'Soil pH Min':[user_input['soil_ph']], 'Soil pH Max':[user_input['soil_ph']],
                'Soil Temp Min (°C)':[user_input['soil_temp']], 'Soil Temp Max (°C)':[user_input['soil_temp']],
                'Rainfall Min (mm/year)':[user_input['rainfall']], 'Rainfall Max (mm/year)':[user_input['rainfall']],
                'Humidity Min (%)':[user_input['humidity']], 'Humidity Max (%)':[user_input['humidity']]
            }
            norm_st=user_input['soil_type'].strip().replace(" ","").lower()
            match_st=next((vt for vt in self.VALID_SOIL_TYPES if norm_st==vt.lower()),
                          user_input['soil_type'].capitalize()) # Find match or default
            input_data['Soil Type']=[match_st]
            input_df=pd.DataFrame(input_data)

            try:
                ordered_cols=self.NUMERIC_FEATURES+self.CATEGORICAL_FEATURES
                input_df=input_df[ordered_cols]
            except KeyError as ke:
                logger.error(f"Input data missing columns: {ke}.")
                return [], None

            probs=self.model.predict_proba(input_df)[0]
            # Ensure classes_ attribute exists and is accessible
            if hasattr(self.model, 'classes_'):
                 classes=self.model.classes_
            elif hasattr(self.model.named_steps['clf'], 'classes_'): # Check classifier step if pipeline
                 classes=self.model.named_steps['clf'].classes_
            else:
                 logger.error("Cannot determine class labels from model.")
                 return [], None


            results=sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
            suggestions=[{"Crop":c, "Confidence":p} for c, p in results if p>=self.MIN_CONFIDENCE_THRESHOLD]

            if not suggestions:
                logger.warning("No crops met threshold.")
                return [], None

            logger.info(f"Generated {len(suggestions)} suggestions.")

            if location:
                weather_data=self.get_weather_data(location)

            weather_risk=self._analyze_weather_risk(weather_data)

            for sug in suggestions:
                market=self.get_market_data(sug['Crop'])
                sug.update(market)
                sug['WeatherRisk']=str(weather_risk)

            return suggestions, weather_data
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}", exc_info=True)
            return [], weather_data # Return potentially fetched weather data even on error


    # --- AI Farming Advice Generation ---
    def generate_farming_advice(self, crop_name: str) -> Optional[str]:
        """Uses AI assistant for detailed farming advice."""
        if not self.chat_assistant:
            logger.warning("AI Chat Assistant unavailable.")
            return "AI Assistant unavailable."

        logger.info(f"Generating farming advice for '{crop_name}' using AI...")
        advice=[]
        prompts=[
            f"Detailed best farming practices for growing {crop_name} (soil prep, planting depth/spacing, water, fertilizer NPK/timing, sun/temp, harvest indicators).",
            f"Common pests AND diseases affecting {crop_name} (describe signs/damage).",
            f"Effective treatments for {crop_name} pests/diseases (organic/cultural, chemical active ingredients, prevention)."
        ]
        sections=["Practices", "Pests & Diseases", "Treatments"]

        for i, prompt in enumerate(prompts):
            s_name=sections[i]
            logger.info(f"Asking AI: {s_name} for {crop_name}...")
            resp=self.chat_with_assistant(prompt)
            advice.append(f"--- {s_name} ---")
            if "I'm sorry" in resp or "API error" in resp:
                logger.warning(f"AI failed {s_name} for {crop_name}: {resp}")
                advice.append(f"(Advice retrieval failed: {resp})")
            else:
                advice.append(resp.strip())
            advice.append("\n")
            time.sleep(1.5) # Delay between heavy AI calls

        final="\n".join(advice)
        if len(final.replace("---","").replace("\n","").strip()) < 100: # Arbitrary check
            logger.error(f"AI advice too short/failed for {crop_name}.")
            return final + "\n\n[Warning: AI response seems incomplete.]"

        return final

    def save_farming_advice(self, advice_text: str, crop_name: str) -> Optional[str]:
        """Saves generated farming advice to file."""
        if not advice_text:
            return None
        safe_cn="".join(c for c in crop_name if c.isalnum() or c in (' ','_')).rstrip().replace(' ','_')
        fname=f"farming_advice_{safe_cn}_{datetime.now().strftime('%Y%m%d')}.txt"
        fpath=os.path.join(os.getcwd(), fname)
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f"🌱 Farming Advice: {crop_name} 🌱\n{'='*(20+len(crop_name))}\n\n{advice_text}")
            logger.info(f"Farming advice saved: {fpath}")
            return fpath
        except IOError as e:
            logger.error(f"Failed save farming advice {fname}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving advice: {e}")
            return None

# --- Main Execution Function ---
def main():
    """Main function to run the crop recommender."""
    print("\n🌱 PROFESSIONAL CROP RECOMMENDER v9.1 (Open-Meteo) 🌱") # Version bump
    print("="*60)
    print("Recommendations with Weather, Market Data, AI Chat & Farming Advice\n")
    try:
        recommender=ProfessionalCropRecommender()

        # Model Loading/Training Check
        if not recommender.model or not recommender.preprocessor:
            print("\n🔄 No valid model found.")
            need_train=True
            if recommender.df is None or recommender.df.empty:
                print("   Attempting data load...")
                need_train = recommender.load_and_merge_data()
                if not need_train:
                    print("\n❌ CRITICAL ERROR: Failed data load. Exiting.")
                    return
            if need_train:
                print("\n🤖 Training new model...")
                model_info=recommender.train_model()
                if not model_info or not recommender.model:
                    print("\n❌ CRITICAL ERROR: Model training failed. Exiting.")
                    return
                print(f"✅ Model trained. Acc: {model_info.get('best_score',0):.1%}. Algo: {model_info.get('classifier_type','N/A')}")
                if model_info.get('best_score',0)<0.65:
                    print("⚠️ Warn: Model accuracy moderate.")
        else:
            print("\n✅ Loaded pre-trained model/data.")

        # Menu Loop
        while True:
            print("\n" + "="*20 + " MENU " + "="*20)
            print("1. Get Recommendations & Advice")
            print("2. Ask General Farming Question (AI)")
            print("3. Retrain Model")
            print("4. Exit")
            print("="*46)
            choice=input("\nChoose option (1-4): ").strip()

            if choice=='1':
                if not recommender.model or not recommender.preprocessor:
                    print("\n❌ Model unavailable. Train first (3).")
                    continue
                print("\n📝 Enter growing conditions:")
                try:
                    # Input loops for validation
                    while True:
                        try:
                            ph=float(input(" - Soil pH (e.g., 6.5): "))
                            break
                        except ValueError: print("   Invalid number.")
                    while True:
                        try:
                            temp=float(input(" - Avg Soil temp (°C) (e.g., 25): "))
                            break
                        except ValueError: print("   Invalid number.")
                    while True:
                        stype=input(f" - Soil type ({', '.join(recommender.VALID_SOIL_TYPES)}): ").strip()
                        if stype.replace(" ","").lower() in [s.lower() for s in recommender.VALID_SOIL_TYPES]:
                            break
                        else: print(f"   Invalid. Choose from list.")
                    while True:
                        try:
                            rain=int(input(" - Annual rainfall (mm) (e.g., 1200): "))
                            break
                        except ValueError: print("   Invalid whole number.")
                    while True:
                        try:
                            hum=float(input(" - Avg humidity (%) (e.g., 70): "))
                            break
                        except ValueError: print("   Invalid number.")

                    u_input={'soil_ph':ph, 'soil_temp':temp, 'soil_type':stype, 'rainfall':rain, 'humidity':hum}
                    recommender.validate_inputs(**u_input) # Final check

                    loc=input(" - Location (City, Country - optional for weather): ").strip() or None

                    print("\n🔍 Analyzing crops & fetching data...")
                    suggs, w_data = recommender.generate_suggestions(u_input, loc)

                    if not suggs:
                        print("\n⚠️ No suitable crops found.")
                        recommender.debug_conditions(u_input)
                        continue

                    m_info={'best_score':0.85, 'best_params':{}} # Dummy
                    if hasattr(recommender.model, 'best_score_'):
                        m_info['best_score'] = recommender.model.best_score_

                    report=recommender.generate_report(suggs, m_info, u_input, loc, w_data)
                    print("\n"+"="*25+" Recommendation Report "+"="*25)
                    print(report)
                    print("="*73)

                    # Save outputs
                    txt_f=recommender.save_report_to_file(report)
                    html_f=recommender.save_report_as_html(report)
                    plot_f=recommender.plot_recommendations(suggs)
                    if txt_f: print(f"\n📄 Text report: {txt_f}")
                    if html_f: print(f"📄 HTML report: {html_f}")
                    if plot_f: print(f"📈 Plot: {plot_f}")

                    # AI Advice
                    top_c=suggs[0]['Crop']
                    print(f"\n🤖 Generating farming advice for '{top_c}' (AI)...")
                    advice=recommender.generate_farming_advice(top_c)
                    if advice:
                        advice_f=recommender.save_farming_advice(advice, top_c)
                        print(f"💡 Farming advice: {advice_f}" if advice_f else "⚠️ Could not save advice.")
                    else:
                        print(f"⚠️ Could not generate advice for {top_c}.")

                except ValueError as ve:
                    print(f"\n❌ Input Error: {ve}")
                except Exception as e:
                    print(f"\n❌ Unexpected error: {e}")
                    logger.error("Error in recommendation option", exc_info=True)

            elif choice=='2':
                if not recommender.chat_assistant:
                    print("\n❌ AI Chat unavailable.")
                    continue
                print("\n💬 Ask general farming question ('back'/'quit' to return):")
                while True:
                    q=input("   Your question: ").strip()
                    if q.lower() in ('exit','quit','back'):
                        break
                    if not q:
                        continue
                    print("\n🤖 Thinking...")
                    resp=recommender.chat_with_assistant(q)
                    print("\n"+"="*15+" AI Response "+"="*15)
                    print(resp)
                    print("="*43+"\n")
                    if input("Save response? (y/n): ").lower()=='y':
                        fname=f"general_advice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        try:
                            with open(fname, 'w', encoding='utf-8') as f:
                                f.write(f"Q: {q}\n\nA:\n{resp}")
                            print(f"✅ Saved: {fname}")
                        except IOError as e:
                            print(f"❌ Error saving: {e}")

            elif choice=='3':
                print("\n🔄 Retraining Model...")
                need_load = (recommender.df is None or recommender.df.empty)
                if need_load:
                    print("   Reloading data...")
                    data_ok = recommender.load_and_merge_data()
                else:
                    data_ok = True # Data already loaded

                if not data_ok:
                    print("\n❌ ERROR: Failed data load for retraining.")
                    continue

                print("   Starting training...")
                m_info=recommender.train_model()
                if m_info and recommender.model:
                    print(f"✅ Retrained. Acc: {m_info.get('best_score',0):.1%}. Algo: {m_info.get('classifier_type','N/A')}")
                    if m_info.get('best_score',0)<0.65:
                        print("⚠️ Warning: Model accuracy moderate after retraining.")
                else:
                    print("\n❌ ERROR: Retraining failed.")

            elif choice=='4':
                print("\nExiting...")
                break
            else:
                print("Invalid choice (1-4).")

    except ValueError as ve:
        print(f"\n❌ Config Error: {ve}")
        logger.error(f"Config Error: {ve}", exc_info=False)
    except ImportError as ie:
        print(f"\n❌ Missing Library: `pip install {ie.name}`")
        logger.error(f"Import Error: {ie}")
    except Exception as e:
        print(f"\n❌ Unexpected critical error: {e}")
        print("   Check 'crop_recommender.log'.")
        logger.error("Critical error in main", exc_info=True)
    finally:
        print("\n🙏 Thank you for using the Crop Recommender!")

# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        import matplotlib
        import seaborn
    except ImportError:
        logger.warning("Matplotlib/Seaborn missing. Plotting disabled.")
    main()

# --- END OF FILE me.py ---