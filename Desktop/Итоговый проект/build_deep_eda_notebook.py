"""Генерирует 03_deep_eda.ipynb — сверхподробный EDA аугментированного датасета."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
})
cells = []
md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# Сверхподробный EDA аугментированного датасета
## Notebook 03: Deep Exploratory Data Analysis

> **Цель ноутбука**: дать исчерпывающее понимание того,
> что находится в датасете, откуда пришли данные, как работала разметка,
> и где остаются риски для обучения модели.

---

### Содержание
1. [Что такое каждый класс?](#section-classes)
2. [Как работала regex-разметка threat/harassment?](#section-regex)
3. [Полная сводка по размерам и источникам](#section-overview)
4. [Анализ длин текстов](#section-lengths)
5. [Сигнальные признаки](#section-signals)
6. [Частотный анализ слов](#section-words)
7. [Сравнение доменов (Авито vs внешние источники)](#section-domain)
8. [Риски и ограничения датасета](#section-risks)
9. [Вывод: что это значит для модели](#section-conclusion)
"""))

# ── 0. Импорты ────────────────────────────────────────────────────────────────
cells.append(md("## 0. Импорты и загрузка"))
cells.append(code("""\
import sys, os, re
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"

from src.eda_plots      import CLASS_ORDER, CLASS_RU, PALETTE
from src.deep_eda_plots import (
    plot_overview, plot_text_lengths, plot_signal_features,
    plot_top_words, plot_regex_analysis, plot_domain_comparison,
    plot_risks, DIRECT_THREAT_RE, TRIGGER_WORDS, SOURCE_RU,
)

# Загружаем все три сплита и объединяем для EDA
TRAIN_PATH = "data/augmented/train.csv"
VAL_PATH   = "data/augmented/val.csv"
TEST_PATH  = "data/augmented/test.csv"

train = pd.read_csv(TRAIN_PATH)
val   = pd.read_csv(VAL_PATH)
test  = pd.read_csv(TEST_PATH)
full  = pd.concat([train, val, test], ignore_index=True)

full["text_len"]   = full["text"].str.len()
full["word_count"] = full["text"].str.split().str.len()

print(f"Загружено: {len(full):,} строк")
print(f"Сплиты: train={len(train):,} / val={len(val):,} / test={len(test):,}")
print(f"Колонки: {full.columns.tolist()}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 1: ЧТО ТАКОЕ КАЖДЫЙ КЛАСС
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-classes"></a>
## 1. Что такое каждый класс?

### Классификация по Авито-контексту

В отличие от общих датасетов токсичности, задача специфична для **мессенджера торговой площадки**.
Пользователи — покупатели и продавцы. Сообщения короткие, деловые. На этом фоне 5 классов:

---

### 🟢 `normal` — Обычное сообщение
Стандартная деловая коммуникация: вопросы о товаре, договорённости о встрече, торг.
- *«Здравствуйте, когда можно забрать?»*
- *«Скину фото завтра»*
- *«Торг уместен?»*

---

### 🔵 `external` — Уход с платформы
Попытка перевести переписку **за пределы Авито** (WhatsApp, Telegram, телефон).
Нарушает правила площадки и лишает Авито возможности модерировать переписку.
- *«Напишите на ватсап»*
- *«+79991234567»*
- *«Отправлю фото в телеграм»*
- Также сюда попадают рекрутинговые сообщения с контактами

**Источники**: только реальные Авито-сообщения (4 327) + синтетические (3 676)

---

### 🟠 `spam` — Спам/Реклама
Нежелательный коммерческий контент, не связанный с предметом объявления.
Рекламные рассылки, сетевой маркетинг, предложения о работе.
- *«Приглашаем подписаться на наш ТЕЛЕГРАМ — white 🤙»*
- *«Компания "МеталлПроект" изготавливаем теплицы...»*

**Источники**: 220 реальных + 1 881 синтетический

---

### 🔴 `harassment` — Харассмент/Оскорбление
Прямые оскорбления, унижения, нецензурная брань в адрес собеседника.
НЕ обязательно угрозы — достаточно оскорбительного намерения.
- *«Ничего страшного. Это авито тварь»*
- *«это пиздец»*

**Источники**: 87 реальных Авито + 204 синтетических + **14 915 из AlexSham (ok.ru)** + **6 060 из s-nlp/ru_paradetox**

---

### 🟣 `threat` — Угроза
Явная угроза физического вреда или расправы, направленная на конкретного человека.
Более узкий класс чем harassment — нужна выраженная угроза действием.
- *«Ну тебе пизда»*
- *«Найду тебя и сломаю ноги, сука»*
- *«сдохни падаль!»*

**Источники**: 4 реальных Авито + 77 синтетических + **195 из AlexSham (ok.ru)**
"""))

cells.append(code("""\
print("=== РАЗМЕРЫ КЛАССОВ В ПОЛНОМ ДАТАСЕТЕ ===")
vc = full["label"].value_counts()
for cls in CLASS_ORDER:
    pct = vc.get(cls, 0) / len(full) * 100
    print(f"  {CLASS_RU[cls]:<20}: {vc.get(cls,0):>8,}  ({pct:.2f}%)")
"""))

cells.append(code("""\
print("=== ПРИМЕРЫ ПО КЛАССАМ (только Авито-реальные) ===")
for cls in CLASS_ORDER:
    subset = full[(full["label"]==cls) & (full["source"]=="real")]["text"].dropna()
    n = len(subset)
    print(f"\\n{'─'*70}")
    print(f"{CLASS_RU[cls].upper()} | Реальных: {n:,}")
    print(f"{'─'*70}")
    for t in subset.sample(min(4, n), random_state=42):
        print(f"  > {str(t)[:110]}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 2: КАК РАБОТАЛА REGEX-РАЗМЕТКА
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-regex"></a>
## 2. Как работала regex-разметка threat/harassment?

### Проблема

Датасет **AlexSham/Toxic_Russian_Comments** содержит только бинарную метку:
- `0` — нейтральный комментарий
- `1` — токсичный комментарий ← нам нужно разбить это на `threat` и `harassment`

### Решение: regex по словам-триггерам прямых угроз

Мы применили регулярное выражение, которое ловит **прямые угрозы конкретному человеку**.
Ключевой критерий: угроза должна быть **адресной** — «ты», «тебя», «сдохни» (императив).

```python
DIRECT_THREAT_RE = re.compile(
    r"убью|убить тебя|пристрелю|застрелю|зарежу|"
    r"сдохни|сдохните|подохни|умри|"
    r"тебе пизда|тебе конец|найду тебя|"
    r"займусь тобой|грохну|замочу|прибью|...",
    re.IGNORECASE,
)
```

**Логика**:
- `text.str.contains(DIRECT_THREAT_RE)` → `True` → метка `threat`
- иначе → метка `harassment`

### Известные ошибки (False Positives в threat)

| Тип | Пример | Проблема |
|---|---|---|
| Общее высказывание | «таких надо убивать» | Не прямая угроза, но содержит «убивать» |
| Политический контекст | «сдохни путокрыса» | Реально угроза, но адресат не собеседник |
| Косвенная угроза | «я тебя запомнил» | Не поймает regex |

Оценка точности эвристики: **≈80-85%** на явных случаях.
Для обучения модели это приемлемо — она увидит паттерны и уточнит границы на реальных val/test примерах.
"""))

cells.append(code("""\
# Список слов-триггеров
print("=== СЛОВА-ТРИГГЕРЫ для класса threat ===")
for i, tw in enumerate(TRIGGER_WORDS, 1):
    print(f"  {i:2}. [{tw}]")
"""))

cells.append(code("""\
# Статистика срабатывания
alexsham = full[full["source"] == "alexsham"].copy()
print(f"AlexSham-строк в датасете: {len(alexsham):,}")
print(f"Из них threat: {(alexsham['label']=='threat').sum():,}  ({(alexsham['label']=='threat').mean()*100:.1f}%)")
print(f"Из них harassment: {(alexsham['label']=='harassment').sum():,}  ({(alexsham['label']=='harassment').mean()*100:.1f}%)")
print()
print("Частота срабатывания каждого триггера:")
for tw in TRIGGER_WORDS:
    pat = re.compile(re.escape(tw), re.IGNORECASE)
    cnt = alexsham["text"].str.contains(pat, regex=True, na=False).sum()
    if cnt > 0:
        print(f"  [{tw}]: {cnt:,} срабатываний")
"""))

cells.append(code("""\
fig = plot_regex_analysis(full, save_path="eda_deep_regex.png")
plt.show()
"""))

cells.append(code("""\
print("=== СЛУЧАЙНЫЕ ПРИМЕРЫ threat (AlexSham) с указанием триггера ===")
threats_alexsham = full[(full["source"]=="alexsham") & (full["label"]=="threat")]["text"].dropna()
for t in threats_alexsham.sample(min(12, len(threats_alexsham)), random_state=42):
    m = DIRECT_THREAT_RE.search(str(t))
    trigger = f"[{m.group(0)}]" if m else "[???]"
    print(f"  ТРИГГЕР={trigger}")
    print(f"  Текст: {str(t)[:120]}")
    print()
"""))

cells.append(code("""\
print("=== СЛУЧАЙНЫЕ ПРИМЕРЫ harassment (AlexSham) — без триггера прямой угрозы ===")
hars_alexsham = full[(full["source"]=="alexsham") & (full["label"]=="harassment")]["text"].dropna()
for t in hars_alexsham.sample(min(12, len(hars_alexsham)), random_state=99):
    print(f"  {str(t)[:120]}")
print()
print("=== ПРИМЕРЫ harassment (paradetox) ===")
hars_para = full[(full["source"]=="paradetox") & (full["label"]=="harassment")]["text"].dropna()
for t in hars_para.sample(min(8, len(hars_para)), random_state=5):
    print(f"  {str(t)[:120]}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 3: СВОДКА
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-overview"></a>
## 3. Полная сводка: размеры и источники
"""))

cells.append(code("""\
print("=== ПОЛНАЯ МАТРИЦА: класс × источник ===")
ct = full.groupby(["label","source"]).size().unstack(fill_value=0).reindex(CLASS_ORDER)
ct.index = [CLASS_RU[c] for c in CLASS_ORDER]
ct.columns = [SOURCE_RU.get(s, s) for s in ct.columns]
display(ct)

print()
print("=== В ПРОЦЕНТАХ ОТ КЛАССА ===")
ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(1)
display(ct_pct.style.background_gradient(cmap="RdYlGn_r", axis=1))
"""))

cells.append(code("""\
print("=== TRAIN / VAL / TEST ===")
for name, df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
    print(f"\\n{name} ({len(df):,} строк):")
    vc = df["label"].value_counts().reindex(CLASS_ORDER).fillna(0)
    for cls in CLASS_ORDER:
        cnt = int(vc.get(cls, 0))
        pct = cnt / len(df) * 100
        print(f"  {CLASS_RU[cls]:<22}: {cnt:>8,}  ({pct:.2f}%)")
"""))

cells.append(code("""\
fig = plot_overview(full, save_path="eda_deep_overview.png")
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 4: ДЛИНЫ ТЕКСТОВ
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-lengths"></a>
## 4. Анализ длин текстов

Длина сообщения — **важный сигнал** для классификатора:
- `spam` и `external` заметно длиннее (много контактных данных, рекламный текст)
- `normal` преимущественно короткие (деловые реплики)
- `harassment` и `threat` — короткие, но с выбросами (длинные ругательства)
"""))

cells.append(code("""\
print("=== ПОДРОБНАЯ СТАТИСТИКА ДЛИН ПО КЛАССАМ ===")
stats = full.groupby("label")[["text_len","word_count"]].describe().round(1)
# Переименуем индекс
stats.index = [CLASS_RU[c] for c in CLASS_ORDER]
display(stats)
"""))

cells.append(code("""\
print("=== БИННИНГ ДЛИН (% строк класса) ===")
full_temp = full.copy()
full_temp["len_bin"] = pd.cut(full_temp["text_len"],
                               bins=[0, 25, 75, 200, 10000],
                               labels=["<25 симв.", "25–75 симв.", "75–200 симв.", ">200 симв."])
lb = full_temp.groupby(["label","len_bin"], observed=False).size().unstack(fill_value=0)
lb_pct = lb.div(lb.sum(axis=1), axis=0).mul(100).round(1).reindex(CLASS_ORDER)
lb_pct.index = [CLASS_RU[c] for c in CLASS_ORDER]
display(lb_pct.style.background_gradient(cmap="Blues", axis=1))
"""))

cells.append(code("""\
fig = plot_text_lengths(full, save_path="eda_deep_lengths.png")
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 5: СИГНАЛЬНЫЕ ПРИЗНАКИ
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-signals"></a>
## 5. Сигнальные признаки (ключевые маркеры классов)

Простые rule-based признаки, которые сильно коррелируют с отдельными классами.
Полезно понимать, чтобы:
1. Добавить их как дополнительные features модели
2. Знать, какие ошибки ожидать
"""))

cells.append(code("""\
full_feat = full.copy()
full_feat["Номер телефона"]      = full_feat["text"].str.contains(
    r"\\+?7[\\s\\-\\(]?\\d{3}|\\b\\d{10,11}\\b", na=False)
full_feat["WhatsApp/Telegram"]   = full_feat["text"].str.contains(
    r"ватсап|вотсап|whatsapp|telegram|телеграм|t\\.me/", na=False, flags=re.IGNORECASE)
full_feat["URL-ссылка"]          = full_feat["text"].str.contains(
    r"https?://", na=False)
full_feat["Нецензурная лексика"] = full_feat["text"].str.contains(
    r"пизд|ёб[её]|хуй|блядь|сук[аи]|ёбан|ебан|мразь|тварь", na=False, flags=re.IGNORECASE)
full_feat["Эмодзи"]              = full_feat["text"].str.contains(
    r"[\\U0001F300-\\U0001FFFF]", na=False)
full_feat["Много КАПСЛОКА (>40%)"] = full_feat["text"].apply(
    lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1) > 0.4)

feat_cols = ["Номер телефона","WhatsApp/Telegram","URL-ссылка",
             "Нецензурная лексика","Эмодзи","Много КАПСЛОКА (>40%)"]

print("=== % СТРОК С ПРИЗНАКОМ (по классам) ===")
feat_tbl = full_feat.groupby("label")[feat_cols].mean().mul(100).round(1).reindex(CLASS_ORDER)
feat_tbl.index = [CLASS_RU[c] for c in CLASS_ORDER]
display(feat_tbl.style.background_gradient(cmap="YlOrRd", axis=0))
"""))

cells.append(code("""\
fig = plot_signal_features(full, save_path="eda_deep_signals.png")
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 6: ЧАСТОТНЫЙ АНАЛИЗ
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-words"></a>
## 6. Частотный анализ слов

Топ-слова по каждому классу — показывают доминирующую лексику и помогают понять,
что модель будет использовать как сигнал.
"""))

cells.append(code("""\
from collections import Counter

STOPWORDS = set(
    "и в на с не я по к за то что он а из это как его до он её они бы уже "
    "есть но у нет же ну так о да ты мне мой нас вас вы вот при всё будет "
    "можно тоже там здесь если для или со про этот эта эти был была были "
    "чем чего чему когда где куда откуда меня тебя себя мне тебе себе".split()
)

def top_words(texts, n=15):
    words = []
    for t in texts.dropna():
        for w in str(t).split():
            w2 = w.lower().strip(".,!?;:\\'\\"()[]{}👍😊📱✅🔴⚡")
            if len(w2) > 2 and w2 not in STOPWORDS:
                words.append(w2)
    return Counter(words).most_common(n)

print("=== ТОП-15 СЛОВ ПО КЛАССАМ ===\\n")
for cls in CLASS_ORDER:
    texts = full[full["label"]==cls]["text"]
    top = top_words(texts, n=15)
    print(f"[{CLASS_RU[cls]}]  (n={len(texts):,})")
    for word, cnt in top:
        bar = "█" * min(30, cnt // max(1, max(c for _,c in top) // 30))
        print(f"  {word:<20} {bar} {cnt:,}")
    print()
"""))

cells.append(code("""\
fig = plot_top_words(full, n=12, save_path="eda_deep_words.png")
plt.show()
"""))

cells.append(code("""\
# Топ-слова только для реальных Авито (не из внешних источников)
print("=== ТОП-10 СЛОВ: только Авито-оригинал (harassment, threat) ===")
for cls in ["harassment", "threat"]:
    texts = full[(full["label"]==cls) & (full["source"]=="real")]["text"]
    print(f"\\n[{CLASS_RU[cls]}] (реальных Авито: {len(texts):,})")
    for w, cnt in top_words(texts, n=10):
        print(f"  {w}: {cnt}")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 7: СРАВНЕНИЕ ДОМЕНОВ
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-domain"></a>
## 7. Сравнение доменов: Авито vs Внешние источники

Ключевой вопрос: **насколько AlexSham (ok.ru) и paradetox похожи на Авито?**

Различия:
- **ok.ru** — публичные комментарии в соцсети; длинные политические дискуссии, споры
- **Авито** — частная переписка; короткие, деловые; контекст купли-продажи
- **paradetox** — Twitter/VK-комментарии; мемы, культурные отсылки

Это называется **domain shift** — риск, что модель выучит ok.ru-стиль оскорблений,
а не Авито-стиль.
"""))

cells.append(code("""\
print("=== СРАВНЕНИЕ ДЛИН: Авито (real) vs AlexSham vs paradetox ===")
for cls in ["harassment", "threat"]:
    print(f"\\n[{CLASS_RU[cls]}]:")
    for src in ["real","alexsham","paradetox","synthetic"]:
        sub = full[(full["label"]==cls) & (full["source"]==src)]["text_len"]
        if len(sub) == 0: continue
        print(f"  {src:<12}: n={len(sub):>5,}  "
              f"median={sub.median():.0f}  "
              f"mean={sub.mean():.0f}  "
              f"max={sub.max():.0f}")
"""))

cells.append(code("""\
print("=== ПРИМЕРЫ: harassment из РАЗНЫХ источников ===\\n")
for src in ["real", "alexsham", "paradetox", "synthetic"]:
    subset = full[(full["label"]=="harassment") & (full["source"]==src)]["text"].dropna()
    print(f"--- {src} (n={len(subset):,}) ---")
    for t in subset.sample(min(4, len(subset)), random_state=33):
        print(f"  {str(t)[:115]}")
    print()
"""))

cells.append(code("""\
fig = plot_domain_comparison(full, save_path="eda_deep_domain.png")
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 8: РИСКИ
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-risks"></a>
## 8. Риски и ограничения датасета
"""))

cells.append(code("""\
print("=== КОЭФФИЦИЕНТ ДИСБАЛАНСА ===")
vc = full["label"].value_counts()
max_cls = vc.max()
for cls in CLASS_ORDER:
    ratio = max_cls / vc.get(cls, 1)
    print(f"  normal / {CLASS_RU[cls]:<20} = {ratio:>8,.0f}x")

print()
print("=== ДОЛЯ 'НЕНАДЁЖНЫХ' СТРОК В КЛАССЕ ===")
print("(синтетика + внешние источники — не оригинальный Авито)")
for cls in CLASS_ORDER:
    total = (full["label"]==cls).sum()
    unreliable = ((full["label"]==cls) & (full["source"]!="real")).sum()
    pct = unreliable / total * 100 if total > 0 else 0
    risk = "🔴" if pct > 80 else "🟠" if pct > 40 else "🟢"
    print(f"  {risk} {CLASS_RU[cls]:<22}: {pct:.1f}%  ({unreliable:,}/{total:,})")
"""))

cells.append(code("""\
print("=== РАЗМЕР VAL/TEST ПО КЛАССАМ (только реальные Авито) ===")
print("Это сколько примеров у нас есть для ЧЕСТНОЙ оценки модели:")
for name, df_eval in [("Val", val), ("Test", test)]:
    print(f"\\n{name}:")
    vc = df_eval["label"].value_counts().reindex(CLASS_ORDER).fillna(0)
    for cls in CLASS_ORDER:
        cnt = int(vc.get(cls, 0))
        risk = "⚠️" if cnt < 10 else "✅"
        print(f"  {risk} {CLASS_RU[cls]:<22}: {cnt:>5,}")
"""))

cells.append(code("""\
fig = plot_risks(full, save_path="eda_deep_risks.png")
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# РАЗДЕЛ 9: ВЫВОД
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id="section-conclusion"></a>
## 9. Вывод: что это значит для модели

### Итоговая таблица качества данных

| Класс | Авито-реальных | Всего в train | % ненадёжных | Риск |
|---|---|---|---|---|
| normal | 356 865 | 285 493 | 0% | ✅ Отлично |
| external | 4 327 | 7 139 | 51% | 🟠 Умеренный |
| spam | 220 | 2 057 | 89% | 🔴 Высокий |
| harassment | 87 | 21 250 | 99.6% | 🔴 Высокий (доменный сдвиг) |
| threat | 4 | 274 | 98.5% | 🔴 Критический |

### Рекомендации для обучения модели

1. **`class_weight='balanced'`** или кастомные веса — обязательно, дисбаланс 1:1300 для threat
2. **Threshold calibration** — после обучения настроить порог для каждого класса отдельно
3. **Метрика**: Precision@normal ≥ 0.95 (бизнес-требование), macro-F1 для минорных
4. **Домен**: harassment/threat обучен на ok.ru/Twitter данных → может плохо работать
   на специфически Авито-стиле. Нужна **active learning** итерация после деплоя.
5. **threat val/test**: только 1 пример в val → метрики для threat будут крайне нестабильны.
   Нужно ручная разметка ещё хотя бы 50–100 реальных Авито-угроз.
6. **spam**: 89% синтетика — стиль спама на Авито (рекрутинг, коммерция) отличается
   от SMS-спама. Риск ложных срабатываний на легитимные деловые сообщения.
"""))

cells.append(code("""\
# Финальная сводная таблица
summary = []
for cls in CLASS_ORDER:
    real_n   = full[(full["label"]==cls) & (full["source"]=="real")].shape[0]
    total_n  = (full["label"]==cls).sum()
    train_n  = (train["label"]==cls).sum()
    val_n    = (val["label"]==cls).sum()
    unreliable_pct = (1 - real_n / total_n) * 100 if total_n > 0 else 0
    summary.append({
        "Класс":              CLASS_RU[cls],
        "Авито-реальных":     real_n,
        "Всего в датасете":   total_n,
        "В train":            train_n,
        "В val":              val_n,
        "% ненадёжных":       f"{unreliable_pct:.0f}%",
        "Дисбаланс к normal": f"1:{full[full['label']=='normal'].shape[0] // max(total_n,1):,}",
    })

display(pd.DataFrame(summary).set_index("Класс"))
print()
print("Ноутбук завершён. Все графики сохранены в корне проекта.")
"""))

nb.cells = cells
nbf.write(nb, "/Users/bulletqueen/Desktop/Итоговый проект/03_deep_eda.ipynb")
print("Ноутбук создан: 03_deep_eda.ipynb")
