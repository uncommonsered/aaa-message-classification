import requests
from dotenv import load_dotenv
import os
from gigachat import GigaChat


load_dotenv()

KEY = os.getenv("AUTHORIZATION_KEY_SBER")
giga = GigaChat(credentials=KEY)

TOKEN = giga.get_token().access_token

SYSTEM_PROMPT_SPAM = """Ты аугментируешь сообщения для датасета модерации.

Для каждого сообщения:
- слегка осовремень текст
- сохрани написание слов с опечатками или намеренными искажением
- добавить Telegram/WhatsApp или @username, t.me
- иногда использовать современный сленг, сокращения, emoji
- немного перефразировать

Нельзя:
- сильно переписывать текст
- менять категорию сообщения
- добавлять одинаковые шаблоны во все сообщения
- добавлять Telegram или ссылки в каждое сообщение

Верни только изменённые сообщения строкой. В качестве разделителя между сообщениями используй символ переноса строки. Не добавляй никаких пояснений, комментариев или текста помимо изменённых сообщений."""


SYSTEM_PROMPT_EXTERNAL_MESSAGES = """Ты аугментируешь сообщения для датасета модерации. Для каждого сообщения необходимо очень аккуратно перефразировать сообщения, не меняя смысла. ТАКЖЕ НЕ МЕНЯЙ
НАЗВАНИЕ МЕССЕНДЖЕРА И ФОРМАТ НОМЕРА ТЕЛЕФОНА, ЕСЛИ ОНИ ЕСТЬ. ОНИ УЖЕ АУГМЕНТИРОВАНЫ И НЕ НУЖДАЮТСЯ В ДОПОЛНИТЕЛЬНОЙ АУГМЕНТАЦИИ.
Каждое сообщение отделено друг от друга символом ;
Разрешается:
    1. Переставлять слова в предложении, сохраняя их смысл.
    2. Заменять слова на синонимы, сохраняя смысл предложения.
    3. Добавлять или удалять незначительные детали, которые не изменяют общий смысл сообщения.
    4. Сохранять все упоминания о внешних контактах (телефоны, карты, почта, ссылки, мессенджеры), при этом обрати СОХРАНИТЬ ИХ НАПИСАНИЕ И ФОРМАТ, НЕ ИЗМЕНЯЯ ИХ. ОНИ УЖЕ АУГМЕНТИРОВАНЫ И НЕ НУЖДАЮТСЯ В ДОПОЛНИТЕЛЬНОЙ АУГМЕНТАЦИИ.
    Верни только изменённые сообщения строкой. В качестве разделителя между сообщениями используй символ ;. Не добавляй никаких пояснений, комментариев или текста помимо изменённых сообщений."""


SYSTEM_PROMPT_HARASSMENT_AND_THREAT = """
Ты фильтруешь сообщения для датасета переписок маркетплейса. Для каждого сообщения нужно определить: может ли сообщение реалистично встретиться
в чате продавца и покупателя на площадке для продажи б/у вещей. Сообщения разделены символом ;

Подходят:
- короткие диалоги
- бытовые оскорбления
- угрозы
- токсичность
- обвинения в мошенничестве
- грубость

Категорически подходят:
- политические комментарии
- обсуждение новостей
- форумный флуд
- мемы
- сообщения содержащие имена, названия компаний, брендов
- длинные рассуждения
- комментарии к видео
- токсичность вне контекста переписки

Ответом верни число от 1 до 5, где 1 - это сообщение, которое точно не может встретиться в чате продавца и покупателя на площадке для продажи б/у вещей, а 5 - это сообщение, 
которое точно может встретиться в таком чате. Не добавляй никаких пояснений, комментариев или текста помимо числа. Числа разделяй символом ;. Числа должны быть целыми и следовать порядку сообщений."""

url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


def update_spam_messages(messages: list[str]) -> list[str]:

    payload = {
        "model": "GigaChat-2",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_SPAM},
            {"role": "user", "content": "\n".join(messages)},
        ],
        "temperature": 0.8,
        "max_tokens": 1000,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    }

    response = requests.post(url, headers=headers, json=payload)

    return response.json()["choices"][0]["message"]["content"].split("\n")


def update_external_messages(messages: list[str]) -> list[str]:

    payload = {
        "model": "GigaChat-2",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EXTERNAL_MESSAGES},
            {"role": "user", "content": ";".join(messages)},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    }

    response = requests.post(url, headers=headers, json=payload)

    return response.json()["choices"][0]["message"]["content"].split(";")


def estimate_harassment_and_threat(messages: list[str]) -> list[int]:

    payload = {
        "model": "GigaChat-2",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_HARASSMENT_AND_THREAT},
            {"role": "user", "content": ";".join(messages)},
        ],
        "temperature": 0,
        "max_tokens": 10,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        return [
            int(x) 
            for x in response.json()["choices"][0]["message"]["content"].split(";")
        ]
    except ValueError:
        # для батча 2 возвращаем минимальные скоры, если модель не захотела генерировать ответ.
        return [1, 1]
