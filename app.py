#! /usr/bin/python3

import random
from config import BOT_CONFIG

"""
Создаем бота для работы с NLP.
1 - Используем правила, тоесть заранее  заготовленные ответы, а четко
    поставленные вопросы. Тоесть четткое правило поведения, на четко
    поставленный intent(намеренье).
2 - Если четкого intent не нашлось, то используем Генеративную Модель
    generative model, которая из больщого числа фраз вернет максимально
    соответствующий ответ.
3 - Если ничего не сработало, то используем заглушки
"""


def get_default_response():
    """Метод Заглушка, возвращает дефолтное значение."""
    candidates = ["Я не понял вас", "Уточните ваш вопрос"]
    return random.choice(candidates)


def generative_answer():
    """Генерация Ответа при помощи Генеративнйо Модели."""
    return None


def get_intent(text):
    """Поиск Интентов, четко поставленых намерений пользователя."""
    # return 'hello'
    return None


# TODO Тут написать код для генерации ответа на четкий интент
def get_response_by_intent(intent):
    """Генерация ответа на сполученный Интент."""
    candidates = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(candidates)


def get_answer(text):
    """Основной метод поиска ответ на поставленный пользователем вопрос"""

    # TODO Ищем Четкое намеренье пользователя.
    intent = get_intent(text)

    if intent:
        return get_response_by_intent(intent)

    # TODO Используем Генеративную модель.
    # response = generative_answer(text)
    response = generative_answer()

    if response:
        return response

    # TODO Если ничего не сработало, то используем метод заглушку.
    return get_default_response()


while True:
    question = input("============ Ввод: ")
    print(get_answer(question))
