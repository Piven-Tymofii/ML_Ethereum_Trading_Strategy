# ML Ethereum Trading Strategy

This repository explores machine learning approaches for generating Ethereum (ETH) trading signals based on price data. It started as a research project to build and evaluate ML pipelines for time-series classification. Later, it was extended into a working Telegram bot that provides real-time trade signals.

## Features

- ✅ Full data pipeline from raw 5-minute ETH price data to engineered features (Technical Indicators)
- 📊 Research notebooks comparing models (LightGBM, CatBoost, Logistic Regression, etc.)
- 🧩 Time Siries CrossVal, automated training and visualization pipelines on the huge training dataset
- 🧪 Signal generation using binary classification (Buy / Sell)  
- 🤖 Deployable Telegram bot for end-user interaction  
- 🐳 Dockerized and scheduled for daily updates + monthly retraining  

## Notebooks

Notebooks are structured for clarity:

- **FETCH DATA API** - first steps in this project; was trying to find an API service that suits the requirements
- **Data Engineering** – feature creation, filtering, and labeling logic  
- **Model Training** – selection, training, and evaluation of classifiers  
- **Final Implementation** – exploring how the project could be deployed  
- **Afterword** – brief notes and reflection  

## Telegram Bot

The bot delivers trading signals via Telegram using the trained model. It fetches new ETH data every day, updates the dataset, and can be extended to support retraining or indicator selection.
You can try it out, it is running on 'render.com' 24/7.
Not financial advice)

Main script: `tg_bot_signals.py`  
Bot: [@ml_eth_signal_bot](https://t.me/ml_eth_signal_bot)

## Setup

```bash
git clone https://github.com/Piven-Tymofii/ML_Ethereum_Trading_Strategy.git
cd ML_Ethereum_Trading_Strategy
pip install -r requirements.txt
```

Configure your `.env` file (API keys, bot token, etc.), and you're ready.

## Deployment

Build & run with Docker:

```bash
docker build -t eth-ml-bot .
docker run --env-file apis.env eth-ml-bot
```

You can use a free cloud service like Render to run the bot 24/7 with scheduled retraining.

## Roadmap

- [x] Initial research & model comparison  
- [x] Bot deployment  
- [ ] Web dashboard or bot analytics  
- [ ] Integrate more advanced signal filtering
- [ ] ...

## License

MIT

-------
