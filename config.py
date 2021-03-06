BOT_CONFIG = {
    "intents": {
        "hello": {
            "examples":  ["Привет", "Добрый день", "Здраствуйте", "Приветики", "Хай"],
            "responses": ["Привет", "Здраствуйте", "Ну привет, челик кью"]
        },
        "goodbye": {
            "examples":  ["Пока", "Всего доброго", "До свидания"],
            "responses": ["Ну пока", "Счастливо"]
        },
        "thanks": {
            "examples":  ["Спасибо", "Спасибо большое!", "Благодарю"],
            "responses": ["Вам спасибо", ""]
        },
        "whatcanyoudo": {
            "examples": ["Что ты умеешь?", "Что можешь?", "Расскажи что умеешь"],
            "responses": ["Отвечать на вопросы. Просто напиши:"]
        },
        "name": {
            "examples": ["Как тебя зовут?", "Как твое имя?", "Имя?", "Твое имя?"],
            "responses": ["Бот, прсото бот, джеймс бот"]
        },
        "weather": {
            "examples":  ["Какая погода в Москве?", "Какая погода?"],
            "responses": ["Погода так себе..."]
        }
    },

    # Результаты заглушки
    "failure_responses": [
        "Я не понял вас",
        "Я не знаю, что ответить",
        "Я вас не понял",
        "Переформулируйте, пожалуйста",
        "Простите я вас не понял",
        "Уточните ваш вопрос",
    ]
}
