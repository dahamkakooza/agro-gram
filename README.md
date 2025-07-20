# agro-gram
Agro-Gram: AI-Driven Crop Farming Solution
Overview
Agro-Gram is an innovative AI-powered agricultural assistant designed to help Ugandan farmers optimize crop selection, improve yields, and reduce post-harvest losses. The system combines machine learning recommendations with real-time weather data, market prices, and an AI chatbot to provide comprehensive farming advice through accessible channels like SMS/USSD.

Features
Core Functionality
Smart Crop Recommendations: Suggests optimal crops based on:

Soil conditions (pH, type, temperature)

Local weather forecasts

Market price trends

AI Farming Assistant: Answers agricultural questions in local languages (Luganda/Runyankole)

Multi-Channel Delivery: Accessible via:

Smartphone app

SMS/USSD for farmers without smartphones

Technical Components
Machine Learning Model: Random Forest and Gradient Boosting classifiers trained on crop datasets

Weather Integration: Open-Meteo API for real-time weather data

Market Intelligence: yFinance integration for commodity prices

AI Chat: Google Gemini-powered agricultural expert system

Installation
Requirements
Python 3.8+

Required packages:

bash
pip install pandas numpy scikit-learn yfinance geopy requests google-generativeai python-dotenv joblib matplotlib seaborn
Setup
Clone the repository

crop.csv (primary dataset)

professional_crops.csv (optional extended dataset)

Usage
Run the main application:

bash
python Agro-gram.py
Menu Options
Get Recommendations: Enter soil/weather conditions to receive crop suggestions

AI Chat: Ask general farming questions

Retrain Model: Update the machine learning model

Exit: Close the application

Project Structure
text
Agro-Gram/
├── Agro-gram.py            # Main application code
├── crop.csv                # Primary crop dataset
├── professional_crops.csv  # Extended crop dataset (optional)
├── .env                    # Configuration file
├── cache/                  # Cache directory
├── pretrained_model.joblib # Saved ML model
├── pretrained_preprocessor.joblib # Data preprocessor
└── crop_prices.json        # Market price database
Documentation
Vision
Triple agricultural productivity by 2040

Reduce post-harvest losses by 40%

Align with Uganda Vision 2040 for agricultural transformation

Target Users
Smallholder farmers

Agricultural extension officers (NAADS)

Agri-business stakeholders

Implementation Strategy
Pilot Phase: 6 months in Masaka District (maize/coffee region)

Key Partners:

Makerere University (soil data)
Google Maps (Location)
MTN Uganda (SMS/USSD integration)

Masaka Farmer Association (market data)
Nakasero Market Vendors Association

Challenges & Solutions
Challenge	Solution
Low tech literacy	Farmer ambassador program
Data gaps	Makerere University partnerships
Market exploitation	Direct price information via USSD
Impact Metrics
Short-term: Train 10,000 farmers, target 30% yield increase

Long-term: $50M added farmer income by 2040

Policy: Data to inform MAAIF subsidy programs

License
Under Process...

Contact
For more information about the Agro-Gram project, please contact the development team.

