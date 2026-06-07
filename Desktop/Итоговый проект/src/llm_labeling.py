"""
Модуль LLM-разметки: переклассификация harassment → threat/harassment
через zero-shot классификацию на HuggingFace моделях.

Модель: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
  - Многоязычная (100+ языков, включая русский)
  - Zero-shot классификация через NLI (Natural Language Inference)
  - Не требует API-ключей — работает локально
  - Размер: ~1.2GB (скачивается один раз, кэшируется HuggingFace)

Задача: из 500 токсичных текстов AlexSham, помеченных regex-эвристикой
как 'harassment', найти реальные угрозы ('threat').

Методология:
  Модель работает хорошо для разделения «насилие/угроза» vs «оскорбление»,
  но классифицирует как угрозу и абстрактные призывы к насилию над третьими
  лицами. Поэтому используется двухэтапный фильтр:
    1. Порог уверенности >= THREAT_THRESHOLD (0.80)
    2. Расширенный regex — должен присутствовать ПРЯМОЙ адресат (ты/вы/имя)
       или явный угрожающий глагол в форме первого лица/повелительного наклонения

  Итог по 500 примерам AlexSham: 3 реальные личные угрозы (0.6%) —
  это подтверждает, что AlexSham — источник оскорблений, а не личных угроз.
"""

import re
import time
import pandas as pd
from pathlib import Path
from typing import Optional

# ── Константы ─────────────────────────────────────────────────────────────────
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
SAVE_EVERY = 50           # сохранять прогресс каждые N примеров
BATCH_SIZE = 32           # размер батча для GPU/CPU инференса

# Метки для zero-shot классификации (на русском — модель понимает)
CANDIDATE_LABELS = [
    "прямая угроза физической расправой над адресатом",
    "оскорбление или ненависть без прямой угрозы насилия",
]

# Шаблон гипотезы для NLI (сильно влияет на точность)
HYPOTHESIS_TEMPLATE = "Этот текст содержит {}."

# Порог уверенности для класса 'threat' (если ниже → harassment)
# 0.80 — консервативный порог: отсекает большинство ложных срабатываний
# Дополнительный фильтр через PERSONAL_THREAT_RE применяется в build_threat_upgrade_df
THREAT_THRESHOLD = 0.80

# ── Regex: прямая личная угроза (второй этап фильтрации) ──────────────────────
# Ловим форму: «[насилие] + [тебя/вас]» или «[тебя/вас] + [насилие]»
# плюс классические прямые формы (убью, сдохни и т.д.)
_ADDR = r"(?:тебя|тебе|вас|вам)"
_VIOL = (
    r"(?:убить|убью|убивать|поубивать|расстрелять|расстрелю|зарежу|зарезать|"
    r"застрелить|застрелю|повесить|повешу|задушить|задушу|придушить|придушу|"
    r"утопить|утоплю|прибить|прибью|грохнуть|грохну|замочить|замочу|"
    r"кастрировать|кастрирую|найти|найду|сломать|сломаю|переломать|"
    r"вырезать|вырежу|отрезать|отрежу|уничтожить|уничтожу|прикончить|прикончу)"
)

PERSONAL_THREAT_RE = re.compile(
    rf"(?:{_ADDR}\s+(?:надо\s+|следует\s+)?{_VIOL})|"   # «тебя расстрелять»
    rf"(?:{_VIOL}\s+{_ADDR})|"                           # «убью тебя»
    r"(?:сдохни|сдохните|подохни|подохните|умри|умрите)\b|"
    r"тебе\s*(?:пизда|хана|конец|капут|кирдык)\b|"
    r"пизда\s+тебе\b|"
    r"найду\s+(?:тебя|вас)|приду\s+за\s+тобой|"
    r"башку\s+(?:снесу|проломлю|разобью)|"
    r"убью\b|пристрелю|застрелю|зарежу|пырну|придушу|задушу|"
    r"займусь\s+тобой|разберусь\s+с\s+тобой|"
    r"грохну\s+(?:тебя|вас)|замочу\s+(?:тебя|вас)|"
    r"прибью\s+(?:тебя|вас)|прикончу\s+(?:тебя|вас)|"
    r"башку\s+(?:снесу|проломлю|разобью)|"
    r"кости\s+переломаю|ноги\s+переломаю|"
    r"вырежу\s+кишки|выпущу\s+кишки|"
    r"найду\s+и\s+убью|найду\s+и\s+накажу",
    re.IGNORECASE,
)


def load_classifier():
    """Загружает zero-shot pipeline. Модель кэшируется HuggingFace."""
    from transformers import pipeline
    print(f"⏳ Загрузка модели {MODEL_NAME}...")
    print("   (первый запуск: скачивается ~1.2GB, далее — из кэша)")
    clf = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    print("✅ Модель загружена!")
    return clf


def classify_text(classifier, text: str) -> tuple[str, float]:
    """
    Классифицирует один текст.
    Returns: ('threat'|'harassment', confidence_score)
    """
    result = classifier(text, CANDIDATE_LABELS)
    # result["labels"][0] — класс с наибольшей вероятностью
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    if "угроза" in top_label and top_score >= THREAT_THRESHOLD:
        return "threat", float(top_score)
    else:
        return "harassment", float(top_score)


def classify_batch(classifier, texts: list[str]) -> list[tuple[str, float]]:
    """Батчевая классификация для ускорения."""
    results = classifier(texts, CANDIDATE_LABELS, batch_size=BATCH_SIZE)
    labels = []
    for result in results:
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        if "угроза" in top_label and top_score >= THREAT_THRESHOLD:
            labels.append(("threat", float(top_score)))
        else:
            labels.append(("harassment", float(top_score)))
    return labels


def run_labeling(
    input_path: str | Path = "data/llm_labeling_sample.csv",
    output_path: str | Path = "data/llm_labeling_sample.csv",
    resume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Запускает zero-shot разметку всех строк в файле.

    resume=True  — пропускаем уже размеченные строки
    resume=False — перезапускаем с нуля
    verbose=True — выводим примеры угроз

    Returns: DataFrame с заполненными метками
    """
    classifier = load_classifier()
    df = pd.read_csv(input_path)

    # Добавляем колонку confidence если нет
    if "llm_confidence" not in df.columns:
        df["llm_confidence"] = None

    # Определяем строки для разметки
    if resume:
        todo_mask = df["llm_label"].isna()
        n_done = (~todo_mask).sum()
        n_todo = todo_mask.sum()
        print(f"\n📋 Всего: {len(df)} | Уже размечено: {n_done} | Осталось: {n_todo}")
    else:
        df["llm_label"] = None
        df["llm_confidence"] = None
        todo_mask = pd.Series([True] * len(df), index=df.index)
        print(f"\n📋 Запускаем с нуля: {len(df)} строк")

    todo_indices = df.index[todo_mask].tolist()

    if not todo_indices:
        print("✅ Все строки уже размечены!")
        return df

    print(f"\n🚀 Начинаем zero-shot классификацию {len(todo_indices)} текстов...")
    print(f"   Модель: {MODEL_NAME}")
    print(f"   Порог для threat: {THREAT_THRESHOLD}")
    print("=" * 60)

    n_threat = 0
    n_harassment = 0
    start_time = time.time()

    # Обрабатываем батчами
    for batch_start in range(0, len(todo_indices), BATCH_SIZE):
        batch_indices = todo_indices[batch_start:batch_start + BATCH_SIZE]
        batch_texts = [df.at[idx, "text"] for idx in batch_indices]

        # Классификация батча
        batch_results = classify_batch(classifier, batch_texts)

        for idx, (label, score) in zip(batch_indices, batch_results):
            df.at[idx, "llm_label"] = label
            df.at[idx, "llm_confidence"] = round(score, 4)

            if label == "threat":
                n_threat += 1
                if verbose:
                    text_preview = df.at[idx, "text"][:70]
                    print(f"  🔴 THREAT [{score:.2f}]: {text_preview}...")

        n_harassment = batch_start + len(batch_indices) - n_threat

        # Прогресс
        processed = batch_start + len(batch_indices)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(todo_indices) - processed) / rate if rate > 0 else 0
        print(f"  [{processed:3d}/{len(todo_indices)}] "
              f"threat={n_threat} | har={processed-n_threat} | "
              f"{rate:.1f} ex/s | ETA={eta:.0f}s")

        # Сохранение прогресса
        if processed % SAVE_EVERY == 0 or processed == len(todo_indices):
            df.to_csv(output_path, index=False)
            print(f"  💾 Сохранено ({processed}/{len(todo_indices)})")

    # Финальное сохранение
    df.to_csv(output_path, index=False)

    elapsed_total = time.time() - start_time
    total_labeled = len(todo_indices)
    print("\n" + "=" * 60)
    print(f"✅ РАЗМЕТКА ЗАВЕРШЕНА за {elapsed_total:.0f}с")
    print(f"   threat:     {n_threat:4d} ({n_threat/total_labeled*100:.1f}%)")
    print(f"   harassment: {total_labeled-n_threat:4d} ({(total_labeled-n_threat)/total_labeled*100:.1f}%)")
    print(f"   Сохранено:  {output_path}")

    return df


def build_threat_upgrade_df(
    labeled_path: str | Path = "data/llm_labeling_sample.csv",
    min_confidence: float = THREAT_THRESHOLD,
    require_personal: bool = True,
) -> pd.DataFrame:
    """
    Возвращает строки, переклассифицированные в 'threat'.

    Двухэтапный фильтр:
      1. Уверенность LLM >= min_confidence (по умолчанию THREAT_THRESHOLD=0.80)
      2. Если require_personal=True — текст должен также совпасть с
         PERSONAL_THREAT_RE (личная угроза с прямым адресатом)

    Вывод: только примеры, прошедшие оба фильтра.
    Эти строки добавляем в train как новые угрозы.
    """
    df = pd.read_csv(labeled_path)

    # Этап 1: фильтр по LLM-метке и уверенности
    mask = df["llm_label"] == "threat"
    if "llm_confidence" in df.columns:
        conf_mask = df["llm_confidence"].fillna(0.0) >= min_confidence
        mask = mask & conf_mask

    threats = df[mask].copy()
    n_after_llm = len(threats)

    # Этап 2: regex-фильтр на прямое личное обращение
    if require_personal:
        personal_mask = threats["text"].str.contains(PERSONAL_THREAT_RE, na=False)
        threats = threats[personal_mask].copy()

    threats["label"] = "threat"
    threats["source"] = "alexsham_llm"
    threats["synthetic"] = 0

    result = threats[["text", "label", "source", "synthetic"]].reset_index(drop=True)

    print(f"📊 LLM предсказал угрозу: {(df['llm_label']=='threat').sum()} / {len(df)}")
    print(f"   После порога {min_confidence:.2f}: {n_after_llm}")
    if require_personal:
        print(f"   После regex-фильтра (личная угроза): {len(result)}")
    print(f"✅ Новых угроз для обучения: {len(result)}")
    return result


def get_labeling_stats(
    labeled_path: str | Path = "data/llm_labeling_sample.csv",
) -> dict:
    """Статистика по результатам разметки."""
    df = pd.read_csv(labeled_path)
    total = len(df)
    done = int(df["llm_label"].notna().sum())
    threat_n = int((df["llm_label"] == "threat").sum())
    harassment_n = int((df["llm_label"] == "harassment").sum())

    return {
        "total": total,
        "labeled": done,
        "pending": total - done,
        "threat": threat_n,
        "harassment": harassment_n,
        "threat_rate": round(threat_n / done * 100, 1) if done > 0 else 0,
        "avg_confidence": round(
            df.loc[df["llm_label"].notna(), "llm_confidence"].mean(), 3
        ) if "llm_confidence" in df.columns and done > 0 else None,
    }
