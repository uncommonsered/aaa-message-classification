import numpy as np
import re

# введем переменные TG_VARIANTS и WA_VARIANTS, содержащие различные варианты написания мессенджеров
TG_VARIANTS = [
    "telegram",
    "telegarm",
    "telegrm",
    "te1egram",
    "t3legram",
    "telegramm",
    "tele gram",
    "tele.grm",
    "tele_grm",
    "телеграм",
    "телеграмм",
    "тeлеграм",
    "телегpам",
    "телегрaм",
    "телегрм",
    "телеграм.",
    "тг",
    "tg",
    "t.g",
    "t_g",
]

WA_VARIANTS = [
    "whatsapp",
    "whatsap",
    "whats app",
    "whats_app",
    "what-sapp",
    "wh4tsapp",
    "whatsupp",
    "whatsaap",
    "watsapp",
    "ватсап",
    "ватцап",
    "вотсап",
    "ват сап",
    "ват_сап",
    "ват-сап",
    "ватцапп",
    "вацап",
    "васап",
    "wa",
    "w.a",
]

# введем словари TG_NEAR и WA_NEAR, которые будут содержать для каждой буквы список букв, которые находятся рядом на клавиатуре и могут быть ошибочно написаны вместо нее
TG_NEAR = {
    "t": ["r", "y"],
    "e": ["w", "r"],
    "l": ["k", "o"],
    "g": ["f", "h"],
    "r": ["e", "t"],
    "a": ["q", "s"],
    "m": ["n", "j"],
    "т": ["е", "ь"],
    "е": ["у", "н"],
    "л": ["д", "о"],
    "г": ["ш", "н"],
    "р": ["к", "т"],
    "а": ["ф", "п"],
    "м": ["ь", "и"],
}

WA_NEAR = {
    "w": ["q", "e"],
    "h": ["g", "j"],
    "a": ["q", "s"],
    "t": ["r", "y"],
    "s": ["a", "d"],
    "p": ["o", "["],
    "o": ["i", "p"],
    "в": ["ф", "а"],
    "а": ["ф", "п"],
    "т": ["е", "ь"],
    "с": ["ы", "в"],
    "ц": ["у", "к"],
    "п": ["з", "р"],
    "о": ["л", "щ"],
}


def augment_phone_number(number: str) -> str:
    """
    Функция для аугментации телефонного номера, которая вставляет различные разделители между цифрами. Она может:
        1. Вставлять различные разделители между цифрами
        2. Заменять некоторые цифры на похожие по написанию символы, такие как '0' на 'O', '1' на 'I', '5' на 'S' и т.д
        3. Заменять некоторые цифры на слова, например '8' на 'восемь'
    """
    way = np.random.choice(
        ["separators", "similar_symbols", "words"], p=[0.8, 0.1, 0.1]
    )

    # вставляем различные разделители между цифрами
    if way == "separators":
        separators = [" ", "", ".", "_", "-", "*", ","]
        augmented_number = ""
        for digit in number:
            augmented_number += digit
            augmented_number += np.random.choice(separators, p = [0.4, 0.3, 0.05, 0.05, 0.1, 0.05, 0.05])

    # заменяем некоторые цифры на похожие по написанию символы
    if way == "similar_symbols":
        similar_symbols = {
            "0": ["о", "o", "O"],
            "1": ["l", "I", "і"],
            "2": ["z", "Z"],
            "3": ["з"],
            "4": ["ч"],
            "5": ["s", "S"],
            "6": ["б"],
            "7": ["7"],
            "8": ["в", "В"],
            "9": ["q", "g"],
        }
        augmented_number = ""
        for digit in number:
            if digit in similar_symbols and np.random.rand() < 0.2:
                augmented_number += np.random.choice(similar_symbols[digit])
            else:
                augmented_number += digit

    # заменяем некоторые цифры на слова
    if way == "words":
        number_to_words = {
            "0": "ноль",
            "1": "один",
            "2": "два",
            "3": "три",
            "4": "четыре",
            "5": "пять",
            "6": "шесть",
            "7": "семь",
            "8": "восемь",
            "9": "девять",
        }
        augmented_number = ""
        for digit in number:
            if digit in number_to_words and np.random.rand() < 0.2:
                augmented_number += number_to_words[digit]
            else:
                augmented_number += digit

    return augmented_number


def generate_phone_number() -> str:
    """
    Функция для генерации случайного телефонного номера (российских мобильных операторов)
    Взяты отсюда - https://www.kody.su/mobile/
    """
    first_digit = np.random.choice(["+7", "8"])
    oprator_codes = np.random.choice(
        list(range(900, 907))
        + list(range(908, 935))
        + list(range(936, 940))
        + list(range(941, 943))
        + [942]
        + list(range(949, 956))
        + [958, 959]
        + list(range(960, 972))
        + list(range(977, 998))
        + [999]
    )
    number = f"{first_digit}{oprator_codes}{np.random.randint(1000000, 9999999)}"
    return number


def generate_phone_number_and_augment() -> str:
    """
    Функция для генерации случайного телефонного номера и его аугментации. Она может:
        1. Вставлять различные разделители между цифрами
        2. Заменять некоторые цифры на похожие по написанию символы, такие как '0' на 'O', '1' на 'I', '5' на 'S' и т.д
        3. Заменять некоторые цифры на слова, например '8' на 'восемь'
    """
    number = generate_phone_number()
    augmented_number = augment_phone_number(number)
    return augmented_number


def augment_phone_number_in_message(message: str) -> str:
    """
    Функция для аугментации телефонного номера в сообщении, которая вставляет различные разделители между цифрами. Она может:
        1. Вставлять различные разделители между цифрами
        2. Заменять некоторые цифры на похожие по написанию символы, такие как '0' на 'O', '1' на 'I', '5' на 'S' и т.д
        3. Заменять некоторые цифры на слова, например '8' на 'восемь'
    """
    phone_pattern = re.compile(
        r"(?<!\d)(?:\+7|8)\s*\(?\d{3}\)?[\s\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)|(?<!\d)\d{11}(?!\d)",
        re.IGNORECASE,
    )

    def augment_match(match):
        number = match.group(0)
        augmented_number = augment_phone_number(number)
        return augmented_number

    augmented_message = phone_pattern.sub(augment_match, message)
    return augmented_message


def augment_messanger_name_in_message(message: str) -> str:
    """
    Функция для аугментации названия мессенджера в сообщении. Она может:
        1. Симулировать ошибки набора, заменяя некоторые буквы на соседние на клавиатуре
        2. Добавлять случайные разделители между буквами
        3. Заменять название на одно из его вариантов написания
    """

    tg_pattern = re.compile(r"\b(?:telegram+|телеграм+|тг)\b", re.IGNORECASE)
    wa_pattern = re.compile(r"\b(?:whatsap+|ватсап+|ватцап+|вотсап+)\b", re.IGNORECASE)

    separators = [" ", "-", ".", "_", "", "*", ","]

    way = np.random.choice(
        ["bag_typing_simulation", "differentiation", "add_separators"],
        p=[0.4, 0.4, 0.2],
    )

    def augment_match(match, near, variants):
        """
        Функция для аугментации найденного названия мессенджера
        """
        word = match.group(0)

        # имитация опечаток
        if way == "bag_typing_simulation":
            augmented = ""
            for char in word:
                if char in near and np.random.rand() < 0.5:
                    augmented += np.random.choice(near[char])
                else:
                    augmented += char
            return augmented

        # добавление случайных разделителей между буквами
        elif way == "add_separators":
            augmented = ""

            for char in word:
                augmented += char

                if np.random.rand() < 0.5:
                    augmented += np.random.choice(separators)
            return augmented

        # замена названия на одно из его вариантов написания
        elif way == "differentiation":
            return np.random.choice(variants)

        return word

    # аугментируем названия мессенджеров в тексте. Если встретились оба, то оба будут аугментированы
    message = tg_pattern.sub(
        lambda m: augment_match(m, TG_NEAR, TG_VARIANTS),
        message,
    )

    message = wa_pattern.sub(
        lambda m: augment_match(m, WA_NEAR, WA_VARIANTS),
        message,
    )

    return message
