"""
Синтез примеров для дефицитных классов (`threat`, `spam`).

Зачем: в исходных данных только 277 угроз — критически мало. AlexSham (этап 5)
показал, что это в основном оскорбления, а не личные угрозы. Без синтетики
модель не учится распознавать класс `threat` устойчиво.

Два бесплатных способа (без API-ключей, без внешних LLM):

1. `generate_template_threats(n)` — шаблонная генерация
   Базовые синтаксические структуры угроз + слот-филл словарём
   глаголов/обращений/интенсификаторов → разнообразие лексики и форм.

2. `noise_augment(texts, n_per_text)` — character-level аугментация
   Опечатки, дубли букв, замены ё/е/й/и, разный регистр, разделители.
   Учит модель устойчивости к естественному «шуму» сообщений.

Этическая рамка: контент создаётся ИСКЛЮЧИТЕЛЬНО как обучающие негативные
примеры для классификатора токсичности. Это стандартная практика обучения
content moderation моделей.
"""

import re
import random
from typing import Optional


# ── Шаблонная генерация ───────────────────────────────────────────────────────

# Глаголы угрозы 1-го лица (я что-то сделаю)
_VERBS_1P = [
    "убью", "прибью", "грохну", "замочу", "придушу", "задушу", "удавлю",
    "зарежу", "пырну", "пристрелю", "застрелю", "повешу", "утоплю",
    "сломаю", "переломаю", "размозжу", "разорву", "выпотрошу",
    "найду", "найду и убью", "найду и накажу", "найду и кончу",
    "прикончу", "кончу", "уничтожу",
]

# Инфинитивы — для конструкций «тебя надо {INF}», «таких как ты надо {INF}»
_VERBS_INF = [
    "убить", "прибить", "грохнуть", "замочить", "придушить", "задушить",
    "удавить", "зарезать", "застрелить", "повесить", "утопить",
    "сломать", "переломать", "размозжить", "разорвать", "выпотрошить",
    "найти", "прикончить", "уничтожить", "расстрелять",
]

# Императивы (тебе/ему сделать что-то)
_IMPERATIVES_2P = [
    "сдохни", "подохни", "умри", "удавись", "повесься",
    "утопись", "сгори", "застрелись", "сдохните", "подохните",
]

# Угрожающие исходы для адресата (тебе будет X)
_HARMS = [
    "конец", "пизда", "хана", "капут", "кирдык", "крышка", "смерть",
]

# Интенсификаторы / обращения
_TARGETS = [
    "тебе", "тебя", "тварь", "сука", "падаль", "ублюдок", "мразь",
    "гнида", "урод", "конченый", "паскуда", "выродок",
]

# Маркеры конца сообщения
_ENDS = ["", "!", "!!", "!!!", ".", ".."]


# Шаблоны с слотами {V} — verb 1p, {I} — imperative, {H} — harm, {T} — target
_TEMPLATES = [
    # Прямая угроза с адресатом
    "я тебя {V}",
    "{V} тебя",
    "{V} тебя {T}",
    "я тебя {V} {T}",
    "сейчас тебя {V}",
    "я тебя нахуй {V}",
    # Императив
    "{I}",
    "{I} {T}",
    "{I} нахуй",
    "пошёл нахуй {I}",
    # Будущее зло адресату
    "тебе {H}",
    "{T}, тебе {H}",
    "тебе скоро {H}",
    "тебе сегодня {H}",
    "ты {T}, тебе {H}",
    # Поиск + расправа
    "найду тебя и {V}",
    "я тебя найду и {V}",
    "приду за тобой и {V}",
    "разберусь с тобой",
    "я с тобой разберусь {T}",
    "я тебя из под земли достану",
    # Безличное «надо» — здесь инфинитив (грамматически корректно)
    "тебя надо {N}",
    "таких как ты надо {N}",
    "{T}, тебя надо {N}",
    # Части тела / органы
    "башку тебе снесу",
    "челюсть тебе сломаю",
    "руки тебе оторву",
    "кости тебе переломаю",
    "ноги тебе переломаю",
    "выпущу тебе кишки",
    # Соц-инжиниринг угрозы
    "знаю где ты живёшь, приду",
    "я знаю твой адрес, скоро увидимся",
    "посмотрим как ты запоёшь когда я приду",
    "ты у меня кровью умоешься",
    "молись чтобы я тебя не нашёл",
]


def generate_template_threats(n: int = 500, seed: int = 42) -> list[str]:
    """
    Генерирует n уникальных примеров угроз через шаблоны + слот-филл.

    Каждый пример — короткая фраза с прямым адресатом ("тебя"/"тебе"/...).
    Результат — список уникальных текстов (дедуплицирован).
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    # Защита от бесконечного цикла, если шаблонов мало
    max_tries = n * 20

    for _ in range(max_tries):
        if len(out) >= n:
            break
        tpl = rng.choice(_TEMPLATES)
        text = tpl.format(
            V=rng.choice(_VERBS_1P),
            N=rng.choice(_VERBS_INF),
            I=rng.choice(_IMPERATIVES_2P),
            H=rng.choice(_HARMS),
            T=rng.choice(_TARGETS),
        )
        text += rng.choice(_ENDS)
        # Лёгкая нормализация
        text = re.sub(r"\s+", " ", text).strip().lower()
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    return out


# ── Character-level augmentation ──────────────────────────────────────────────

# Типичные «шумы» русского интернет-сленга
_CYR_HOMOGLYPHS = {
    "е": "ё", "ё": "е",
    "и": "й", "й": "и",
    "о": "0", "о": "о",   # дубликат — игнор, демонстрирует, что 0/о редко
}

# Раскладочные опечатки (соседи на QWERTY)
_NEIGHBORS = {
    "а": "вы", "в": "ач", "ы": "цф", "о": "лж", "л": "од",
    "р": "пк", "к": "ре", "е": "кн", "н": "ег", "т": "иь",
    "и": "тм", "с": "чм", "м": "ис", "п": "ра",
}


def _swap_neighbor(text: str, rng: random.Random, prob: float = 0.05) -> str:
    """С вероятностью prob заменяет символ на соседа по клавиатуре."""
    out = []
    for ch in text:
        if rng.random() < prob and ch in _NEIGHBORS:
            out.append(rng.choice(_NEIGHBORS[ch]))
        else:
            out.append(ch)
    return "".join(out)


def _duplicate_char(text: str, rng: random.Random, prob: float = 0.05) -> str:
    """С вероятностью prob дублирует букву (тыыы)."""
    out = []
    for ch in text:
        out.append(ch)
        if rng.random() < prob and ch.isalpha():
            out.append(ch)
    return "".join(out)


def _drop_char(text: str, rng: random.Random, prob: float = 0.03) -> str:
    """С вероятностью prob удаляет букву (пропуск)."""
    return "".join(ch for ch in text if rng.random() >= prob or not ch.isalpha())


def _yo_e(text: str, rng: random.Random) -> str:
    """Случайно заменяет ё↔е (естественный шум русских текстов)."""
    if rng.random() < 0.5:
        return text.replace("ё", "е")
    return text


def _case_noise(text: str, rng: random.Random) -> str:
    """Случайные регистровые варианты (все строчные / Первая Заглавная / ВСЁ КАПС)."""
    r = rng.random()
    if r < 0.2:
        return text.upper()
    if r < 0.4:
        return text.capitalize()
    return text  # по умолчанию строчные


def noise_augment(
    texts: list[str],
    n_per_text: int = 2,
    seed: int = 42,
) -> list[str]:
    """
    Создаёт n_per_text шумных вариантов каждого текста через char-level операции.

    Цель: модель видит «угрозу» в разных формах (с опечатками, дублями букв,
    разным регистром, без ё). Учит устойчивость, а не запоминание лексики.
    """
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set(texts)

    for orig in texts:
        for _ in range(n_per_text * 3):  # × 3 для запаса при коллизиях
            if sum(1 for x in out if x.startswith(orig[:6])) >= n_per_text:
                break
            t = orig
            t = _yo_e(t, rng)
            t = _swap_neighbor(t, rng, prob=0.04)
            t = _duplicate_char(t, rng, prob=0.04)
            t = _drop_char(t, rng, prob=0.02)
            t = _case_noise(t, rng)
            t = re.sub(r"\s+", " ", t).strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


# ── Сборка ────────────────────────────────────────────────────────────────────

# Регекс «явная личная угроза» — тот же фильтр, что в src/llm_labeling.py
# Используется чтобы НЕ тиражировать мусор: в train у класса `threat`
# попадаются неуместные примеры (длинный спам, политические призывы), которые
# шумовая аугментация только размножит.
PERSONAL_THREAT_RE = re.compile(
    r"(?:тебя|тебе|вас|вам)\s+(?:надо\s+)?(?:убить|убью|расстрелять|зарезать|повесить|задушить|найти|кастрировать)|"
    r"(?:убить|убью|расстрелять|зарезать|повесить|задушить)\s+(?:тебя|тебе|вас|вам)|"
    r"(?:сдохни|сдохните|подохни|подохните|умри|умрите)\b|"
    r"тебе\s*(?:пизда|хана|конец|капут|кирдык)\b|"
    r"найду\s+(?:тебя|вас)|приду\s+за\s+тобой|"
    r"башку\s+(?:снесу|проломлю|разобью)|"
    r"убью\b|пристрелю|застрелю|зарежу|пырну|придушу|задушу",
    re.IGNORECASE,
)

# Длина — короткие сообщения мессенджера; режем явный спам/политику
MIN_THREAT_LEN = 5
MAX_THREAT_LEN = 120


def filter_clean_threats(texts: list[str]) -> list[str]:
    """
    Из произвольного списка оставляет только «чистые» личные угрозы:
    подходящая длина + совпадение с PERSONAL_THREAT_RE.
    """
    out = []
    for t in texts:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not (MIN_THREAT_LEN <= len(s) <= MAX_THREAT_LEN):
            continue
        if PERSONAL_THREAT_RE.search(s):
            out.append(s)
    return out


def build_synthetic_threats(
    n_template: int = 500,
    n_noise_per_original: int = 2,
    original_threats: Optional[list[str]] = None,
    seed: int = 42,
) -> list[str]:
    """
    Полный пайплайн: шаблонная генерация + шумовая аугментация *очищенных* существующих.

    original_threats: если передать список реальных меток `threat`, к ним сначала
    применяется `filter_clean_threats` (выбрасывает спам/политику/длинные),
    потом char-level шум. Это убирает риск тиражирования мусорной разметки.

    Returns: список уникальных синтетических угроз (после дедупликации).
    """
    out: list[str] = []
    out.extend(generate_template_threats(n_template, seed=seed))
    if original_threats:
        clean = filter_clean_threats(original_threats)
        out.extend(noise_augment(clean, n_per_text=n_noise_per_original, seed=seed))
    # Дедуп по нижнему регистру
    seen = set()
    unique = []
    for t in out:
        key = t.lower().strip()
        if key in seen or not key:
            continue
        seen.add(key)
        unique.append(t)
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
# Spam-генератор
# ═══════════════════════════════════════════════════════════════════════════════

# Лексика для шаблонов spam
_SPAM_PRODUCTS = [
    "айфон", "смартфон", "ноутбук", "планшет", "наушники", "часы",
    "куртка", "кроссовки", "сумка", "косметика", "духи", "телевизор",
    "холодильник", "стиральная машина", "пылесос", "ковёр",
    "бытовая техника", "автомобиль", "велосипед", "электросамокат",
]
_SPAM_CATEGORIES = [
    "одежды", "электроники", "косметики", "обуви", "мебели", "техники",
    "детских товаров", "товаров для дома", "автомобилей", "украшений",
]
_SPAM_PRICE_LOW  = ["1000", "1500", "2000", "990", "499", "299"]
_SPAM_PRICE_HIGH = ["50000", "100000", "200000", "500000", "1000000"]
_SPAM_PERCENTS = ["30", "40", "50", "60", "70", "80", "90"]
_SPAM_MESSENGERS = ["тг", "telegram", "телеграм", "вотсап", "whatsapp", "вайбер", "viber"]
_SPAM_PHONES = ["+7 (999) ", "+7 (901) ", "8 (495) ", "+7-911-", "8-800-"]
_SPAM_URLS = ["bit.ly/xa9", "tinyurl.com/promo", "clck.ru/ssss", "vk.cc/abcdef"]
_SPAM_PRIZES = ["айфон", "1 млн рублей", "автомобиль", "путёвку", "macbook"]
_SPAM_INCOMES = ["5000", "10000", "30000", "100000", "200000"]
_SPAM_CTA = ["переходи", "пиши", "звони", "регистрируйся", "оставь заявку", "подписывайся"]
_SPAM_URGENCY = [
    "только сегодня", "только сейчас", "до конца недели",
    "успей", "осталось 3 места", "акция ограничена",
]
_SPAM_INTROS = [
    "здравствуйте,", "добрый день,", "приветствую,", "привет!",
    "уважаемые клиенты,", "внимание!", "",
]

_SPAM_TEMPLATES = [
    # Реклама товара
    "{INTRO} распродажа {CAT} со скидкой {PCT}%! {URGENCY}",
    "{INTRO} {PRODUCT} всего за {PRICE} руб! {URGENCY}",
    "купи {PRODUCT} за {PRICE} р, доставка бесплатно",
    "{PRODUCT} новый в упаковке, {PRICE} р, торг возможен. {CTA} в {MSG}",
    "продаю {PRODUCT}, цена договорная, {CTA} в {MSG} {PHONE}",
    # Призывы перейти / связаться
    "пишите в {MSG}, отвечу быстро",
    "переходи по ссылке {URL}",
    "{CTA} на {URL} — там все условия",
    "наш канал в {MSG} — {URL}",
    "звони {PHONE}, расскажем всё",
    # Заработок
    "зарабатывай от {INCOME} рублей в день, {CTA} в {MSG}",
    "пассивный доход {INCOME} в месяц, без вложений",
    "работа на дому, {INCOME}+ р/день, {CTA}",
    "ищем сотрудников, з/п от {INCOME}, гибкий график",
    # Скам / лотереи
    "вы выиграли {PRIZE}! {CTA} для получения",
    "поздравляем! ваш номер выбран — {PRIZE} ваш",
    "вам пришёл подарок: {PRIZE}. {CTA} в {MSG}",
    "розыгрыш {PRIZE} среди подписчиков канала {URL}",
    # Финансовые услуги
    "кредит без отказа за 5 минут, ставка от {PCT}%",
    "займ до {PRICE_H} р, без справок, без отказа",
    "помощь в получении кредита, одобрение 99%",
    # Прочее коммерческое
    "ставки на спорт, бонус за регистрацию",
    "приглашаем в наш {MSG} канал, скидки {PCT}%",
    "промокод на скидку {PCT}% — пиши в личку",
    "лучшие цены на {CAT}, {URL}",
    # Реклама услуг
    "массаж от {PRICE} р, выезд по городу. {PHONE}",
    "клининг, уборка квартир от {PRICE} р",
    "ремонт квартир под ключ, скидки до {PCT}%",
    "репетитор, индивидуальный подход, {PRICE} р/час",
]


def generate_template_spam(n: int = 1000, seed: int = 42) -> list[str]:
    """
    Шаблонная генерация спама в Авито-стиле: реклама, скам, заработок,
    призывы перейти в мессенджер / по ссылке.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    max_tries = n * 25

    for _ in range(max_tries):
        if len(out) >= n:
            break
        tpl = rng.choice(_SPAM_TEMPLATES)
        try:
            text = tpl.format(
                INTRO=rng.choice(_SPAM_INTROS),
                PRODUCT=rng.choice(_SPAM_PRODUCTS),
                CAT=rng.choice(_SPAM_CATEGORIES),
                PRICE=rng.choice(_SPAM_PRICE_LOW),
                PRICE_H=rng.choice(_SPAM_PRICE_HIGH),
                PCT=rng.choice(_SPAM_PERCENTS),
                MSG=rng.choice(_SPAM_MESSENGERS),
                PHONE=rng.choice(_SPAM_PHONES) + "".join(str(rng.randint(0,9)) for _ in range(7)),
                URL=rng.choice(_SPAM_URLS),
                PRIZE=rng.choice(_SPAM_PRIZES),
                INCOME=rng.choice(_SPAM_INCOMES),
                CTA=rng.choice(_SPAM_CTA),
                URGENCY=rng.choice(_SPAM_URGENCY),
            )
        except KeyError:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_synthetic_spam(
    n_template: int = 1000,
    n_noise_per_original: int = 1,
    original_spam: Optional[list[str]] = None,
    seed: int = 42,
) -> list[str]:
    """
    Полный пайплайн для spam: шаблонные примеры + char-noise на существующих.
    """
    out = []
    out.extend(generate_template_spam(n_template, seed=seed))
    if original_spam:
        # Фильтр по длине, чтобы не тянуть огромные тексты
        short = [s for s in original_spam if isinstance(s, str) and 5 <= len(s.strip()) <= 200]
        out.extend(noise_augment(short, n_per_text=n_noise_per_original, seed=seed))
    seen = set()
    unique = []
    for t in out:
        key = t.lower().strip()
        if key in seen or not key:
            continue
        seen.add(key)
        unique.append(t)
    return unique
