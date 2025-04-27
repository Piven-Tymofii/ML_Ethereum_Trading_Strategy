
---

# ML Ethereum Trading Strategy

This repository explores machine learning approaches for generating Ethereum (ETH) trading signals based on price data. It started as a research project to build and evaluate ML pipelines for time-series classification. Later, it was extended into a working Telegram bot that provides real-time trade signals.

## Features

- ✅ Full data pipeline from raw 5-minute ETH price data to engineered features (Technical Indicators)  
- 📊 Research notebooks comparing models (LightGBM, CatBoost, Logistic Regression, etc.)  
- 🧩 Time Series Cross-Validation, automated training, and visualization pipelines on large datasets  
- 🧪 Signal generation using binary classification (Buy / Sell)  
- 🤖 Deployable Telegram bot for end-user interaction  
- 🐳 Dockerized and scheduled for daily updates + monthly retraining  
- 📈 Runs on Microsoft Azure cloud platform  

## Project Stack

| Layer | Technology / Library |
|:-----:|:---------------------|
| **Backend** | Python (pandas, requests, scikit-learn, matplotlib, seaborn, LightGBM, CatBoost, Hugging Face Hub) |
| **Bot Framework** | Python-Telegram-Bot |
| **Infrastructure** | Docker, Docker Hub, Microsoft Azure (Container Instance, Blob Storage) |
| **Model Management** | Hugging Face Hub (skops format) |
| **Scheduling** | APScheduler (daily updates, cron jobs) |
| **Logging** | Python logging module |
| **APIs** | CoinGecko API (ETH data), Telegram API |
| **Cloud Deployment** | Azure App Services, Render.com (alternative option) |

## Notebooks

The research notebooks are structured for clarity:

- **01-Fetch Data** – Initial exploration and evaluation of crypto data providers  
- **02-Data Engineering** – Feature engineering, technical indicators, data filtering, and signal labeling  
- **03-Building ML Models** – Selection, training, evaluation, and tuning of machine learning models  
- **04-Final_Implementation** – Planning the full system for deployment and user interaction  
- **05-Afterword** – Reflections and key lessons learned from the project  

## Telegram Bot

The Telegram bot delivers trading signals using the trained machine learning models.  
It operates by:

- Fetching fresh ETH 5-minute price data  
- Engineering new technical features on the fly  
- Predicting trading signals in real-time  
- Calculating trading statistics for users on demand  

This bot represents just one way of integrating ML-based trading signals into a user-oriented product — the potential for extension (e.g., adding backtesting, portfolio management, alerts) is very large.
I am definitely interested in developing a "2.0" version of the app, but I want to first collect enough data to retrain the models. As noted in 'notebooks/Fetch_Data_Api.ipynb,' there is currently no free way to obtain a sufficient amount of real-time cryptocurrency data, unless there is a recently updated dataset available on Kaggle.

> **You can try it out! It's running 24/7 on Microsoft Azure.**  
> *Note: currently running on a free $100 student credit. After 3 months, maintenance/migration decisions will be made.*  

**Main bot script**: `tg_bot_signals.py`  
**Bot link**: [@ml_eth_signal_bot](https://t.me/ml_eth_signal_bot)

## Setup

```bash
git clone https://github.com/Piven-Tymofii/ML_Ethereum_Trading_Strategy.git
cd ML_Ethereum_Trading_Strategy
pip install -r requirements.txt
```

Configure your `.env` file with your API keys and tokens:

- `COINGECKO_API_KEY`
- `BOT_API_KEY`
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_CONTAINER_NAME`
- `HF_TOKEN_HGB`
- `HF_TOKEN_LGBM`

## Deployment

Build and run the project with Docker:

```bash
docker build -t eth-ml-bot .
docker run --env-file apis.env eth-ml-bot
```

You can also deploy the bot on a cloud platform (e.g., Render.com, Azure App Service) for 24/7 uptime with scheduled retraining.

## Roadmap

- [x] Initial research & model comparison  
- [x] Bot deployment  
- [ ] Add web dashboard for analytics  
- [ ] Integrate more advanced signal filtering and ensemble strategies  
- [ ] Add optional backtesting module  
- [ ] Explore alternative cloud hosting solutions  

## License

MIT License

---