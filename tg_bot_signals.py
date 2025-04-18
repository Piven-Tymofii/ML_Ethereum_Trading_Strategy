import pandas as pd
import requests
from skops.io import get_untrusted_types, load
import os
from datetime import datetime
import pytz
from huggingface_hub import hf_hub_download
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler

# === Load trained models ===

# I uploaded them to the hugging face (best-practice)
model_path_hgb = hf_hub_download(
    repo_id="TymofiiP/HistGradientBoostingClassifier_ETHSignals",
    filename="final_model_hgb.skops",
    use_auth_token=os.getenv("HF_TOKEN_HGB")
)

trusted_types_hgb = [
    "sklearn._loss.link.Interval",
    "sklearn._loss.link.LogitLink",
    "sklearn._loss.loss.HalfBinomialLoss",
    "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

model_hgb = load(model_path_hgb, trusted=trusted_types_hgb)

model_path_lgbm = hf_hub_download(
    repo_id="TymofiiP/LGBMClassifier_ETHSignals",
    filename="final_model_lgbm.skops",
    use_auth_token=os.getenv("HF_TOKEN_LGBM")
)

trusted_types_lgbm = [
    "lightgbm.basic.Booster", 
    'collections.OrderedDict', 
    'lightgbm.sklearn.LGBMClassifier'
]

model_lgbm = load(model_path_lgbm, trusted=trusted_types_lgbm)

# === API keys ===
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
BOT_API_KEY = os.getenv('BOT_API_KEY')

# === Get ETH 5-min data from CoinGecko ===
def get_eth_data():
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=1"
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": COINGECKO_API_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception("API request failed")

    def convert_to_df(key):
        data = response.json()
        df = pd.DataFrame(data[key], columns=["timestamp", key])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("timestamp")

    prices_df = convert_to_df("prices")
    volumes_df = convert_to_df("total_volumes")

    df = prices_df.join([volumes_df])
    df.columns = ["price_usd", "volume_usd"]
    return df

# === Add indicators ===
def add_technical_indicators(df):
    for period in [8, 21]:
        df[f'EMA_{period}'] = df['price_usd'].ewm(span=period, adjust=False).mean()
    for period in [50, 200]:
        df[f'SMA_{period}'] = df['price_usd'].rolling(window=period, min_periods=1).mean()

    def calculate_rsi(df, period=14):
        delta = df['price_usd'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False, min_periods=1).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    for period in [10, 14, 30]:
        df[f'RSI_{period}'] = calculate_rsi(df, period)

    for period in [10, 14]:
        df[f'ROC_{period}'] = df['price_usd'].pct_change(periods=period) * 100

    df['Momentum_14'] = df['price_usd'].diff(14)

    df.dropna(inplace=True)
    df = df.drop(['volume_usd'], axis=1)
    df = df.rename(columns={'price_usd': 'Close'})
    return df

# === Prediction logic ===
def predict_signals(df):
    predictions_hgb = model_hgb.predict(df)
    predictions_lgbm = model_lgbm.predict(df)
    df["Prediction_lgbm"] = predictions_lgbm
    df["Prediction_hgb"] = predictions_hgb
    return df

# === Keyboard UI ===
def get_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Trade", callback_data='trade')],
        [InlineKeyboardButton("❓ Help", callback_data='help')],
        [InlineKeyboardButton("❌ Close", callback_data='close')]
    ])

# === Bot commands ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to ETH ML Signal Bot! 👋", reply_markup=get_keyboard())

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "🤖 *ETH ML Signal Bot Help*\n\n"
        "This bot uses two machine learning models (LightGBM and HistGradientBoosting) "
        "to analyze the latest Ethereum (ETH) price data and predict buy/sell signals.\n\n"
        "📈 *Trade* — Get prediction based on price only\n"
        "❌ *Close* — Remove message"
    )

    if update.message:
        await update.message.reply_text(help_text, parse_mode='Markdown', reply_markup=get_keyboard())
    elif update.callback_query:
        await update.callback_query.message.reply_text(help_text, parse_mode='Markdown', reply_markup=get_keyboard())

# === Handle trade button ===
async def handle_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.edit_message_text("🔍 Fetching latest ETH data and generating predictions...")

    df = get_eth_data()
    df = add_technical_indicators(df)
    df = predict_signals(df)

    # Convert index to Vilnius time
    vilnius_tz = pytz.timezone("Europe/Vilnius")
    df.index = df.index.tz_localize('UTC').tz_convert(vilnius_tz)

    last_28 = df.tail(28)
    msg = "📊 *Last 2 hour of ETH Predictions (28 5-min candles)* — _Timestamps in Vilnius time (UTC+3)_\n\n"
    for index, row in last_28.iterrows():
        time_str = index.strftime('%H:%M')
        msg += (
            f"🕒 {time_str} | "
            f"Close: `${row.Close:.2f}` | "
            f"LGBM: *{'Buy' if row.Prediction_lgbm == 1 else 'Sell'}*, "
            f"HGB: *{'Buy' if row.Prediction_hgb == 1 else 'Sell'}*\n"
        )

    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=msg,
        parse_mode='Markdown',
        reply_markup=get_keyboard()
    )

# === Button handler ===
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "trade":
        await handle_trade(update, context)
    elif query.data == "help":
        await help_command(update, context)
    elif query.data == "close":
        await query.delete_message()

# === Daily data saver ===
def save_daily_data():
    df = get_eth_data()
    os.makedirs("DATA", exist_ok=True)
    df.to_csv("DATA/eth_data_coinGeck.csv", mode='a', header=not os.path.exists("DATA/eth_data_coinGeck.csv"))
    print(f"[+] Daily ETH data saved at {datetime.now()}")

# === Run bot ===
def run_bot():
    print("Starting the bot...")
    app = ApplicationBuilder().token(BOT_API_KEY).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_callback))

    scheduler = BackgroundScheduler()
    scheduler.add_job(save_daily_data, 'interval', days=1)
    scheduler.start()

    print("Bot is polling...")
    app.run_polling(poll_interval=3, timeout=10)

if __name__ == "__main__":
    run_bot()
