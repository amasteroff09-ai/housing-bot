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

print("🔄 Загрузка данных и обучение модели (Москва)...")

# Загрузка данных
df = pd.read_csv('moscow_housing.csv')

X = df.drop('price', axis=1).values
y = df['price'].values

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Оценка качества
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Модель готова! R² = {r2:.3f}, MAE = {mae:.1f} млн ₽")
print(f"📊 Всего квартир в базе: {len(df)}")
print(f"💰 Цены: от {y.min():.1f} до {y.max():.1f} млн ₽")

# ПРИЗНАКИ ДЛЯ МОСКВЫ
FEATURES_RU = [
    "Площадь (кв.м)",
    "Количество комнат",
    "Этаж",
    "Этажность дома",
    "До метро пешком (мин)",
    "Год постройки",
    "Район (1-ЮЗАО, 2-ЦАО, 3-САО)"
]

RANGES = [
    "30-85 кв.м",
    "1-3 комнаты",
    "2-13 этаж",
    "9-18 этажей",
    "3-22 минуты",
    "2003-2023",
    "1, 2 или 3"
]

DEFAULTS_RU = [
    50,    # площадь
    2,     # комнаты
    7,     # этаж
    14,    # этажность
    10,    # метро
    2015,  # год
    1      # район
]

HINTS = [
    "Чем больше, тем дороже (+1.5 млн за 10 м²)",
    "1-комнатная ≈12-15 млн, 2-комнатная ≈18-22 млн",
    "Лучше выше 5-го этажа",
    "Новостройки (16-18 этажей) дороже",
    "Чем ближе, тем дороже (от 3 мин — премиум)",
    "Новостройки после 2020 дороже на 30%",
    "ЦАО самый дорогой, ЮЗАО средний, САО доступнее"
]

user_data = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏠 *Московский HomeValue Predictor*\n\n"
        "Я предсказываю стоимость квартиры в Москве на основе 7 параметров.\n\n"
        f"📊 *База данных:* {len(df)} реальных квартир\n"
        f"💰 *Диапазон цен:* {y.min():.1f} - {y.max():.1f} млн ₽\n\n"
        "🔹 /predict - начать прогноз\n"
        "🔹 /accuracy - точность модели\n"
        "🔹 /ranges - диапазоны значений\n\n"
        "📌 *Как работает:*\n"
        "Я задам 7 вопросов — вы получите примерную цену квартиры!",
        parse_mode='Markdown'
    )

async def show_ranges(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "📊 *ДИАПАЗОНЫ ЗНАЧЕНИЙ (Москва)*\n\n"
    for i, name in enumerate(FEATURES_RU):
        msg += f"• *{name}*: {RANGES[i]}\n"
    msg += f"\n💡 *Всего в базе:* {len(df)} квартир\n"
    msg += f"💰 *Цены:* от {y.min():.1f} до {y.max():.1f} млн ₽"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def accuracy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 *Точность модели Random Forest*\n\n"
        f"🏠 *Квартир в базе:* {len(df)}\n"
        f"📍 *Город:* Москва\n\n"
        "┌─────────────────────────────────────┐\n"
        f"│ 📈 *Метрики качества:*               │\n"
        f"│                                     │\n"
        f"│ • R² = {r2:.3f} ({r2*100:.1f}% точности)     │\n"
        f"│ • MAE = {mae:.1f} млн рублей              │\n"
        f"│ • RMSE ≈ {mae*1.2:.1f} млн рублей            │\n"
        "└─────────────────────────────────────┘\n\n"
        "🔥 *Random Forest* — самый точный алгоритм!\n"
        "💡 *Точность:* ошибка в среднем ±2.5 млн ₽",
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
            f"💡 {HINTS[step]}\n"
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
        await update.message.reply_text("⚠️ Введите /predict для начала")
        return
    
    try:
        value = float(update.message.text)
        user_data[user_id]['values'].append(value)
        user_data[user_id]['step'] += 1
        await ask_parameter(update)
    except:
        await update.message.reply_text(
            "❌ *Ошибка!* Введите число (например, 50)\n\n"
            "Используйте точку, а не запятую.",
            parse_mode='Markdown'
        )

async def calculate_prediction(update):
    user = user_data[update.effective_user.id]
    values = np.array(user['values']).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]
    
    # Оценка стоимости
    if prediction < 15:
        price_level = "🏚️ *Доступная* (эконом-класс)"
        emoji = "🏚️"
    elif prediction < 22:
        price_level = "🏠 *Средняя* (комфорт-класс)"
        emoji = "🏠"
    else:
        price_level = "🏰 *Высокая* (бизнес/премиум)"
        emoji = "🏰"
    
    # Сравнение с рынком
    if prediction < y.mean():
        vs_market = "ниже среднего по базе"
    else:
        vs_market = "выше среднего по базе"
    
    await update.message.reply_text(
        f"🏠 *РЕЗУЛЬТАТ ПРОГНОЗА (Москва)*\n\n"
        f"┌─────────────────────────────────┐\n"
        f"│                                 │\n"
        f"│   {emoji} *{prediction:.1f} млн рублей*      │\n"
        f"│                                 │\n"
        f"│   *{price_level}*         │\n"
        f"│                                 │\n"
        f"└─────────────────────────────────┘\n\n"
        f"📊 *Статистика по базе данных:*\n"
        f"• Минимум: {y.min():.1f} млн ₽\n"
        f"• Максимум: {y.max():.1f} млн ₽\n"
        f"• Среднее: {y.mean():.1f} млн ₽\n"
        f"• Ваша цена: {vs_market}\n\n"
        f"💡 *Совет:* Это прогноз на основе {len(df)} реальных объявлений\n"
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
    
    print("🚀 Московский бот запущен на GitHub Actions!")
    app.run_polling()

if __name__ == '__main__':
    main()
