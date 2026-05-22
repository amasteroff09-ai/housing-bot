import os
import logging
import numpy as np
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BOT_TOKEN = os.environ.get("BOT_TOKEN")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

print("🔄 Загрузка данных...")

df = pd.read_csv('moscow_housing.csv')
X = df.drop('price', axis=1).values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Модель готова! R² = {r2:.3f}, MAE = {mae:.1f} млн ₽")

FEATURES_RU = [
    "Площадь (кв.м)",
    "Количество комнат",
    "Этаж",
    "Этажность дома (всего 24)",
    "До метро пешком (мин)",
    "Год постройки",
    "Район (1-ЦАО, 2-ЗАО/ЮЗАО, 3-спальные)"
]

RANGES = [
    "10-80 кв.м",
    "1-5 комнат",
    "1-24 этаж",
    "24 этажа",
    "2-25 минут",
    "2005-2023",
    "1, 2 или 3"
]

DEFAULTS_RU = [50, 2, 12, 24, 10, 2015, 3]

user_data = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏠 *Московский HomeValue Predictor*\n\n"
        "Предсказываю стоимость квартиры в Москве.\n\n"
        "🔹 /predict - начать прогноз\n"
        "🔹 /accuracy - точность модели\n"
        "🔹 /ranges - диапазоны значений\n\n"
        "📌 *Цены в Москве:*\n"
        "• 1-комнатная: от 8 млн ₽\n"
        "• 2-комнатная: от 9 млн ₽\n"
        "• 3-комнатная: от 9.7 млн ₽\n"
        "• 4-комнатная: от 16 млн ₽\n"
        "• 5-комнатная: от 24 млн ₽",
        parse_mode='Markdown'
    )

async def show_ranges(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "📊 *Диапазоны значений*\n\n"
    for i, name in enumerate(FEATURES_RU):
        msg += f"• *{name}*: {RANGES[i]}\n"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def accuracy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 *Точность модели*\n\n"
        f"• R² = {r2:.3f} ({r2*100:.1f}%)\n"
        f"• Ошибка = ±{mae:.1f} млн ₽\n\n"
        f"📊 *База:* {len(df)} квартир\n"
        f"💰 *Цены:* от {y.min():.1f} до {y.max():.1f} млн ₽",
        parse_mode='Markdown'
    )

async def predict_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data[update.effective_user.id] = {'step': 0, 'values': []}
    await ask_parameter(update)

async def ask_parameter(update):
    user = user_data[update.effective_user.id]
    step = user['step']
    if step < len(FEATURES_RU):
        await update.message.reply_text(
            f"📝 *{FEATURES_RU[step]}*\n\n"
            f"📊 *Диапазон:* {RANGES[step]}\n"
            f"📌 *Пример:* {DEFAULTS_RU[step]}\n\n"
            f"Введите значение:",
            parse_mode='Markdown'
        )
    else:
        await calculate_prediction(update)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_data:
        await update.message.reply_text("Введите /predict")
        return
    
    try:
        value = float(update.message.text)
        user_data[user_id]['values'].append(value)
        user_data[user_id]['step'] += 1
        await ask_parameter(update)
    except:
        await update.message.reply_text("❌ Ошибка! Введите число")

async def calculate_prediction(update):
    user = user_data[update.effective_user.id]
    values = np.array(user['values']).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]
    
    await update.message.reply_text(
        f"🏠 *Прогноз стоимости:*\n\n"
        f"💰 *{prediction:.1f} млн рублей*\n\n"
        f"Введите /predict для нового прогноза",
        parse_mode='Markdown'
    )
    del user_data[update.effective_user.id]

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("accuracy", accuracy))
    app.add_handler(CommandHandler("ranges", show_ranges))
    app.add_handler(CommandHandler("predict", predict_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Бот запущен!")
    app.run_polling()

if __name__ == '__main__':
    main()
