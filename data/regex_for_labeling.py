import re
import requests

# regex patterns for external_messages (phone numbers, emails, urls, car number plates)
phone_pattern = re.compile(
    r"(?<!\d)(?:\+7|8)\s*\(?\d{3}\)?[\s\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)|(?<!\d)\d{11}(?!\d)",
    re.IGNORECASE,
)
card_pattern = re.compile(r"(?<!\d)(?:\d{4}[\s\-]?){3}\d{4}(?!\d)", re.IGNORECASE)
email_pattern = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", re.IGNORECASE
)
url_pattern = re.compile(
    r"\b(?!(?:https?://)?(?:www\.)?avito\.ru\b)(?:https?://|www\.)\S+\b", re.IGNORECASE
)
tg_pattern = re.compile(r"(?:@[\w_]{5,}|t\.me/[\w_]+)", re.IGNORECASE)
msg_pattern = re.compile(
    r"\b(?:telegram+|телеграм+|whatsap+|ватсап+|ватцап+|viber|вайбер|vk|вконтакте)\b",
    re.IGNORECASE,
)

# set for harrassment (swear words, insults)
url_to_russian_bag_words = (
    "https://raw.githubusercontent.com/bars38/Russian_ban_words/master/words.txt"
)
text = requests.get(url_to_russian_bag_words).text
not_valid_word_indices = [
    0, 1, 32, 47, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 109, 110, 111, 114, 
    115, 116, 180, 202, 222, 233, 234, 240, 251, 285, 335, 336, 341, 342, 343, 344, 345, 
    346, 378, 379, 381, 382, 389, 395, 408, 409, 410, 417, 419, 420, 477, 509, 513, 516, 
    520, 618, 619, 629, 880, 886, 939, 1056, 1065, 1072, 1073, 1077, 1111, 1112, 1113, 
    1114, 1119, 1161, 1162, 1168, 1169, 1170, 1221, 1304, 1313, 1314
]

harrassment_set = set(
    [
        text.splitlines()[i]
        for i in range(len(text.splitlines()))
        if i not in not_valid_word_indices
    ]
)


def contains_spam(text: str) -> bool:
    return any(
        [
            url_pattern.search(text),
            tg_pattern.search(text),
            msg_pattern.search(text),
        ]
    )


def contains_harrassment(text: str) -> bool:
    words = re.findall(r"\b\w+\b", text.lower())
    return any(word in harrassment_set for word in words)


def contains_external_message(text: str) -> bool:
    return any(
        [
            phone_pattern.search(text),
            card_pattern.search(text),
            email_pattern.search(text),
            url_pattern.search(text),
            tg_pattern.search(text),
            msg_pattern.search(text),
        ]
    )
