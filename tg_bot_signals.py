import pandas as pd
import requests
from skops.io import get_untrusted_types, load
import os
from datetime import datetime
import pytz
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from huggingface_hub import hf_hub_download
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

# === Setup logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Load Azure Storage ===
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

# Log Azure configuration
logger.info(f"Azure connection string (first 10 chars): {AZURE_CONNECTION_STRING[:10] if AZURE_CONNECTION_STRING else 'None'}")
logger.info(f"Azure container name: {AZURE_CONTAINER_NAME}")

# === In-memory storage for today's predictions ===
today_predictions = pd.DataFrame()

# === Load trained models ===
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
        raise Exception(f"API request failed with status {response.status_code}")

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

# === Calculate trade statistics ===
def calculate_trade_statistics(df):
    if df is None or df.empty:
        return None, None, 0, 0  # Return 0 for total_trades and positive_trades

    trades = []
    current_trade = None

    # Analyze trades for LGBM model
    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['Prediction_lgbm']
        price = row['Close']
        timestamp = row.name

        if current_trade is None:
            current_trade = {
                'start_time': timestamp,
                'start_price': price,
                'signal': signal
            }
        elif signal != current_trade['signal']:
            # End current trade
            price_change = (price - current_trade['start_price']) / current_trade['start_price'] * 100
            is_positive = (price_change > 0 and current_trade['signal'] == 1) or (price_change < 0 and current_trade['signal'] == 0)
            trades.append({
                'start_time': current_trade['start_time'],
                'end_time': timestamp,
                'signal': 'Buy' if current_trade['signal'] == 1 else 'Sell',
                'price_change_pct': price_change,
                'is_positive': is_positive
            })
            # Start new trade
            current_trade = {
                'start_time': timestamp,
                'start_price': price,
                'signal': signal
            }

    # Close last trade if exists
    if current_trade and len(df) > 1:
        last_row = df.iloc[-1]
        price_change = (last_row['Close'] - current_trade['start_price']) / current_trade['start_price'] * 100
        is_positive = (price_change > 0 and current_trade['signal'] == 1) or (price_change < 0 and current_trade['signal'] == 0)
        trades.append({
            'start_time': current_trade['start_time'],
            'end_time': last_row.name,
            'signal': 'Buy' if current_trade['signal'] == 1 else 'Sell',
            'price_change_pct': price_change,
            'is_positive': is_positive
        })

    # Deduplicate trades based on start_time, end_time, and signal
    trade_tuples = {(t['start_time'], t['end_time'], t['signal'], t['price_change_pct'], t['is_positive']) for t in trades}
    trades = [{'start_time': t[0], 'end_time': t[1], 'signal': t[2], 'price_change_pct': t[3], 'is_positive': t[4]} for t in trade_tuples]

    # Calculate statistics
    positive_trades = sum(1 for trade in trades if trade['is_positive'])
    total_trades = len(trades)
    positive_ratio = positive_trades / total_trades if total_trades > 0 else 0

    # Get top 3 positive trades by price change
    positive_trades_list = [t for t in trades if t['is_positive']]
    top_trades = sorted(positive_trades_list, key=lambda x: abs(x['price_change_pct']), reverse=True)[:3]

    return positive_ratio, top_trades, total_trades, positive_trades

# === Keyboard UI ===
def get_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Trade", callback_data='trade')],
        [InlineKeyboardButton("📊 Statistics", callback_data='stats')],
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
        "📊 *Statistics* — View last 24 hours trade success ratio and top 3 trades\n"
        "❌ *Close* — Remove message"
    )

    if update.message:
        await update.message.reply_text(help_text, parse_mode='Markdown', reply_markup=get_keyboard())
    elif update.callback_query:
        await update.callback_query.message.reply_text(help_text, parse_mode='Markdown', reply_markup=get_keyboard())

# === Handle trade button ===
async def handle_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global today_predictions
    query = update.callback_query
    await query.edit_message_text("🔍 Fetching latest ETH data and generating predictions...")

    try:
        df = get_eth_data()
        df = add_technical_indicators(df)
        df = predict_signals(df)
    except Exception as e:
        logger.error(f"Error fetching or processing data: {e}")
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="⚠️ Error fetching data. Please try again later.",
            parse_mode='Markdown',
            reply_markup=get_keyboard()
        )
        return

    # Update in-memory predictions
    today = datetime.now().date()
    df_today = df[df.index.date == today]
    if not df_today.empty:
        if today_predictions.empty or not all(col in today_predictions.columns for col in ['Close', 'Prediction_lgbm', 'Prediction_hgb']):
            today_predictions = df_today
            logger.info("Initialized or reset today_predictions")
        else:
            # Deduplicate more explicitly
            today_predictions = pd.concat([today_predictions, df_today]).reset_index().drop_duplicates(subset=['timestamp']).set_index('timestamp')
            logger.info("Appended new predictions to today_predictions")

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

# === Handle statistics button ===
async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.edit_message_text("🔍 Fetching trade statistics for the last 24 hours...")

    try:
        positive_ratio, top_trades, total_trades, positive_trades = calculate_trade_statistics(today_predictions)
    except Exception as e:
        logger.error(f"Error in calculate_trade_statistics: {e}")
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="⚠️ Error calculating statistics. Please try again later.",
            parse_mode='Markdown',
            reply_markup=get_keyboard()
        )
        return

    if positive_ratio is None:
        msg = "⚠️ No trade data available for the last 24 hours."
    else:
        msg = (
            f"📊 *Last 24 Hours Trade Statistics (LGBM Model)* — _Vilnius time (UTC+3)_\n\n"
            f"✅ *Success Ratio*: {positive_ratio:.2%} ({positive_trades}/{total_trades} trades correct)\n\n"
            f"🏆 *Top 3 Successful Trades*:\n"
        )
        for i, trade in enumerate(top_trades, 1):
            start_time = trade['start_time'].strftime('%H:%M')
            end_time = trade['end_time'].strftime('%H:%M')
            msg += (
                f"{i}. {trade['signal']} from {start_time} to {end_time}\n"
                f"   Price Change: {trade['price_change_pct']:.2f}%\n"
            )
        if not top_trades:
            msg += "No successful trades to display."

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
    elif query.data == "stats":
        await handle_stats(update, context)
    elif query.data == "help":
        await help_command(update, context)
    elif query.data == "close":
        await query.delete_message()

# === Daily data saver ===
def save_daily_data():
    try:
        df = get_eth_data()
        today_str = datetime.now().strftime("%Y-%m-%d")
        blob_name = f"eth_data_{today_str}.csv"
        csv_data = df.to_csv(index=True).encode('utf-8')
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(csv_data, overwrite=True)
        logger.info(f"ETH data uploaded to Azure: {blob_name}")
    except Exception as e:
        logger.error(f"Error saving daily data to Azure: {e}")

# === Reset today_predictions daily ===
def reset_today_predictions():
    global today_predictions
    today_predictions = pd.DataFrame()
    logger.info("Reset today_predictions for new day")

# === Run bot ===
def run_bot():
    logger.info("Starting the bot...")
    app = ApplicationBuilder().token(BOT_API_KEY).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_callback))

    scheduler = BackgroundScheduler(timezone="Europe/Vilnius")
    scheduler.add_job(save_daily_data, CronTrigger(hour=7, minute=0, timezone="Europe/Vilnius"))
    scheduler.add_job(reset_today_predictions, CronTrigger(hour=0, minute=0, timezone="Europe/Vilnius"))
    scheduler.start()
    logger.info("Daily ETH data saving and predictions reset scheduler started")

    logger.info("Bot is polling...")
    app.run_polling(poll_interval=3, timeout=10)

if __name__ == "__main__":
    run_bot()