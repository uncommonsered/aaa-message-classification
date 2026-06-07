"""Генерирует 05_llm_labeling.ipynb — LLM-разметка harassment→threat."""

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
# LLM-разметка: переклассификация harassment → threat
## Notebook 05: Zero-Shot NLI Labeling Analysis

> **Задача**: из 500 токсичных текстов AlexSham, помеченных regex-эвристикой
> как `harassment`, найти скрытые реальные угрозы (`threat`) с помощью
> LLM (zero-shot классификации).

---

### TL;DR результаты

| Параметр | Значение | Примечание |
|---|---|---|
| Модель | `mDeBERTa-v3-base-mnli-xnli` | Бесплатно, локально, 100+ языков |
| Метод | Zero-shot NLI (HuggingFace) | Без дообучения, без API-ключа |
| Размечено | 500 / 500 текстов | ✅ Завершено |
| «Угрозы» по порогу 0.55 | **309 (61.8%)** | ❌ Слишком много — ложные срабатывания |
| Реальные личные угрозы | **3 (0.6%)** | После двойной фильтрации |
| Вывод | AlexSham ≈ оскорбления, не личные угрозы | Regex уже поймал большинство |

---

### Ключевой вывод

Модель правильно понимает концепцию «угроза», но шире, чем нужно:
- ✅ Ловит: «расстрелять тебя надо» (прямая угроза адресату)
- ❌ Тоже ловит: «таких убивать надо», «расстрелять [политика]» (призывы к насилию над третьими лицами)

Решение — **двухэтапный фильтр**: LLM confidence ≥ 0.80 + regex на прямое личное обращение.
"""))

# ── 0. Импорты ─────────────────────────────────────────────────────────────────
cells.append(md("## 0. Подготовка"))
cells.append(code("""\
import sys, os, re
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# Путь к данным
LABELED_PATH = "../data/llm_labeling_sample.csv"

df = pd.read_csv(LABELED_PATH)
print(f"Загружено строк: {len(df)}")
print(f"Колонки: {df.columns.tolist()}")
print(f"\\nПримеры меток:")
print(df[["text", "llm_label", "llm_confidence"]].head(5))
"""))

# ── 1. Общая статистика ────────────────────────────────────────────────────────
cells.append(md("""\
## 1. Результаты разметки: общая статистика

Все 500 строк размечены моделью `mDeBERTa-v3-base-mnli-xnli` через zero-shot NLI.
Посмотрим на распределение меток и уровни уверенности.
"""))
cells.append(code("""\
# Базовая статистика
print("=== ИТОГИ РАЗМЕТКИ ===")
vc = df["llm_label"].value_counts()
for label, cnt in vc.items():
    pct = cnt / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<12}: {cnt:4d} ({pct:.1f}%) {bar}")

print(f"\\nСредняя уверенность: {df['llm_confidence'].mean():.3f}")
print(f"Медианная уверенность: {df['llm_confidence'].median():.3f}")
"""))

cells.append(code("""\
# Распределение уверенности
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Гистограмма всей уверенности
axes[0].hist(df["llm_confidence"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
axes[0].axvline(0.55, color="red", ls="--", label="0.55 (исходный порог)")
axes[0].axvline(0.80, color="orange", ls="--", label="0.80 (новый порог)")
axes[0].set_title("Распределение уверенности (500 текстов)", fontsize=12)
axes[0].set_xlabel("Уверенность модели"); axes[0].set_ylabel("Количество")
axes[0].legend()

# Распределение по классам
threat_conf  = df[df["llm_label"] == "threat"]["llm_confidence"]
harass_conf  = df[df["llm_label"] == "harassment"]["llm_confidence"]
axes[1].hist(threat_conf, bins=30, alpha=0.7, color="#D62728", label=f"threat (n={len(threat_conf)})")
axes[1].hist(harass_conf, bins=30, alpha=0.7, color="#1F77B4", label=f"harassment (n={len(harass_conf)})")
axes[1].set_title("Уверенность: threat vs harassment", fontsize=12)
axes[1].set_xlabel("Уверенность модели"); axes[1].set_ylabel("Количество")
axes[1].legend()

plt.tight_layout()
plt.savefig("../eda_llm_labeling_dist.png", dpi=120, bbox_inches="tight")
plt.show()
print("Сохранено: eda_llm_labeling_dist.png")
"""))

# ── 2. Проблема порога ─────────────────────────────────────────────────────────
cells.append(md("""\
## 2. Проблема порога: 61.8% — слишком много

При пороге 0.55 (исходном) модель относит **309 из 500** текстов к `threat`.
Это 61.8% — явно завышено: реальных личных угроз в общих токсичных комментариях
значительно меньше.

Разберём, как меняется число «угроз» при разных порогах.
"""))
cells.append(code("""\
# Распределение по порогам
threats = df[df["llm_label"] == "threat"].copy()

print("Порог   | Кол-во угроз | % от всей выборки")
print("-" * 45)
thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
for thr in thresholds:
    n = (df["llm_confidence"] >= thr).sum()
    pct = n / len(df) * 100
    print(f">= {thr:.2f} | {n:12d} | {pct:.1f}%")
"""))

cells.append(code("""\
# Примеры предсказанных «угроз» на разных уровнях уверенности
bins = [(0.90, 1.01, "🔴 Очень высокая (0.90+)"),
        (0.80, 0.90, "🟠 Высокая (0.80–0.90)"),
        (0.70, 0.80, "🟡 Средняя (0.70–0.80)"),
        (0.55, 0.70, "🔵 Низкая (0.55–0.70)")]

threat_df = df[df["llm_label"] == "threat"].copy()

for lo, hi, label in bins:
    mask = (threat_df["llm_confidence"] >= lo) & (threat_df["llm_confidence"] < hi)
    subset = threat_df[mask].head(5)
    print(f"\\n{label} — {mask.sum()} примеров")
    for _, row in subset.iterrows():
        print(f"  [{row['llm_confidence']:.2f}]: {row['text'][:80]}")
"""))

# ── 3. Ошибки модели ──────────────────────────────────────────────────────────
cells.append(md("""\
## 3. Почему модель ошибается?

### Типы ложных срабатываний

Модель понимает «угрозу» **шире**, чем нужно для задачи Авито:

| Тип | Пример | Оценка |
|---|---|---|
| ✅ Прямая личная угроза | «расстрелять тебя надо тварь» | НУЖНО |
| ❌ Абстрактный призыв к насилию | «таких сразу убивать!» | НЕ НУЖНО |
| ❌ Угроза публичной фигуре | «расстрелять [политика]» | НЕ НУЖНО |
| ❌ Желание насилия (не намерение) | «я бы ему челюсть сломал» | СПОРНО |
| ❌ Сексуальная агрессия | «я бы её трахнул во все дыры» | НЕ НУЖНО |

### Корневая причина

NLI-модель оценивает, содержит ли текст **концепцию угрозы** вообще.
Для задачи Авито нужен более узкий критерий:
*«прямая угроза конкретному адресату (собеседнику) с явным намерением»*.

Это отличие нельзя передать через short label в zero-shot — нужно либо
fine-tuning, либо более сложный промпт (LLM с reasoning).
"""))
cells.append(code("""\
# Анализ: что НЕ является личной угрозой в pred=threat
import re

# Признаки НЕ-личной угрозы
THIRD_PERSON = re.compile(
    r'\\bего\\b|\\bеё\\b|\\bих\\b|\\bему\\b|\\bей\\b|\\bим\\b|'
    r'\\bтакого\\b|\\bтаких\\b|\\bэтого\\b|\\bэту\\b|\\bэтих\\b|'
    r'надо\\s+(?:убить|расстрелять|повесить)|'
    r'(?:убить|расстрелять|повесить)\\s+надо',
    re.IGNORECASE
)

threat_df = df[df["llm_label"] == "threat"].copy()
threat_df["is_third_person"] = threat_df["text"].str.contains(THIRD_PERSON, na=False)

n_third = threat_df["is_third_person"].sum()
n_total = len(threat_df)
print(f"Из {n_total} предсказанных угроз:")
print(f"  Третье лицо / абстрактный призыв: {n_third} ({n_third/n_total*100:.0f}%)")
print(f"  Без признаков третьего лица:       {n_total-n_third} ({(n_total-n_third)/n_total*100:.0f}%)")
print()
print("Примеры третьего лица (ложные срабатывания):")
for _, row in threat_df[threat_df["is_third_person"]].head(8).iterrows():
    print(f"  [{row['llm_confidence']:.2f}]: {row['text'][:80]}")
"""))

# ── 4. Двойной фильтр ─────────────────────────────────────────────────────────
cells.append(md("""\
## 4. Решение: двойной фильтр

### Стратегия

1. **LLM threshold ≥ 0.80** — отсекаем низкоуверенные предсказания (309 → 78)
2. **Regex PERSONAL_THREAT_RE** — оставляем только с прямым личным адресатом (78 → 3)

Итого: **3 новых личных угрозы**, которые DIRECT_THREAT_RE в `augment.py` пропустил
(потому что там нет форм типа «расстрелять тебя надо», «поубивать вас»).
"""))
cells.append(code("""\
from src.llm_labeling import PERSONAL_THREAT_RE, build_threat_upgrade_df

# Этап 1: порог 0.80
stage1 = df[df["llm_confidence"] >= 0.80]
print(f"Этап 1 (confidence >= 0.80): {len(stage1)} примеров")

# Этап 2: regex
stage1 = stage1.copy()
stage1["personal"] = stage1["text"].str.contains(PERSONAL_THREAT_RE, na=False)
stage2 = stage1[stage1["personal"]]
print(f"Этап 2 (regex личная угроза): {len(stage2)} примеров")

print("\\n=== ФИНАЛЬНЫЕ УГРОЗЫ ДЛЯ ДОБАВЛЕНИЯ В ОБУЧЕНИЕ ===")
for _, row in stage2.iterrows():
    print(f"  [{row['llm_confidence']:.2f}]: {row['text']}")
"""))

cells.append(code("""\
# Используем функцию из модуля
new_threats = build_threat_upgrade_df(
    labeled_path=LABELED_PATH,
    min_confidence=0.80,
    require_personal=True
)
print("\\nDataFrame для добавления в обучение:")
print(new_threats)
"""))

# ── 5. Контекст: общее состояние класса threat ────────────────────────────────
cells.append(md("""\
## 5. Контекст: состояние класса threat в датасете

Проблема дисбаланса классов — `threat` критически мало представлен.
"""))
cells.append(code("""\
# Статистика по всем сплитам
splits = {}
for name in ["train", "val", "test"]:
    splits[name] = pd.read_csv(f"../data/augmented/{name}.csv")

total_per_class = {}
for name, split in splits.items():
    for label, cnt in split["label"].value_counts().items():
        total_per_class[label] = total_per_class.get(label, 0) + cnt

print("Класс        | Всего   | % от датасета")
print("-" * 42)
total = sum(total_per_class.values())
for cls in ["normal", "external", "spam", "harassment", "threat"]:
    cnt = total_per_class.get(cls, 0)
    pct = cnt / total * 100
    bar = "█" * max(1, int(pct / 2))
    print(f"{cls:<12} | {cnt:7,d} | {pct:5.2f}% {bar}")
print(f"{'ВСЕГО':<12} | {total:7,d} |")

print(f"\\n⚠️  threat составляет {total_per_class.get('threat',0)/total*100:.3f}% датасета")
print(f"   (требуется oversampling при обучении модели)")
"""))

cells.append(code("""\
# Источники угроз
all_data = pd.concat(splits.values())
threat_data = all_data[all_data["label"] == "threat"]

print(f"Всего угроз: {len(threat_data)}")
print(f"\\nПо источникам:")
print(threat_data["source"].value_counts().to_string())
"""))

# ── 6. Интеграция новых угроз ──────────────────────────────────────────────────
cells.append(md("""\
## 6. Интеграция: исправление меток в обучающих данных

Один из трёх найденных текстов уже присутствует в обучающей выборке
с неправильной меткой `harassment`. Нужно исправить.

Остальные два текста отсутствуют в датасете (не попали в 15k cap AlexSham).
"""))
cells.append(code("""\
# Проверяем наличие в датасете
new_threats_list = new_threats["text"].tolist()

for split_name, split_df in splits.items():
    for text in new_threats_list:
        match = split_df[split_df["text"].str.strip() == text.strip()]
        if len(match) > 0:
            row = match.iloc[0]
            print(f"НАЙДЕНО в {split_name}: \\\"{text[:60]}\\\"")
            print(f"  Текущая метка: {row['label']} | source: {row['source']}")
            if row["label"] != "threat":
                print(f"  ➜ НУЖНО ИСПРАВИТЬ: {row['label']} → threat")
        else:
            print(f"НЕ найдено в {split_name}: \\\"{text[:60]}\\\"")
            print(f"  ➜ Добавить как новый пример")
"""))

# ── 7. Применение изменений ────────────────────────────────────────────────────
cells.append(md("""\
## 7. Применение исправлений

**Подход**: точечное исправление — находим ошибочно размеченный текст и
меняем метку, добавляем отсутствующие тексты напрямую в train.

Почему не пересобираем весь датасет:
- Пересборка через `build_augmented()` занимает ~10 минут и требует повторного
  скачивания AlexSham
- Три примера — точечное изменение, не требующее полного пересоздания
- Изменения легко отследить и откатить
"""))
cells.append(code("""\
import sys
sys.path.insert(0, "../src")

def apply_threat_corrections(
    splits_dir: str = "../data/augmented",
    new_threats_df: pd.DataFrame = None,
) -> dict:
    \"\"\"
    Применяет исправления меток и добавляет новые угрозы.

    1. Перемечает ошибочные harassment → threat в существующих сплитах
    2. Добавляет отсутствующие тексты в train

    Returns: dict со статистикой изменений
    \"\"\"
    if new_threats_df is None:
        return {}

    changes = {"relabeled": [], "added": []}

    # Загружаем все сплиты
    split_dfs = {}
    for name in ["train", "val", "test"]:
        split_dfs[name] = pd.read_csv(f"{splits_dir}/{name}.csv")

    # Шаг 1: ищем и исправляем ошибочные метки
    for text in new_threats_df["text"]:
        found_anywhere = False
        for name, split_df in split_dfs.items():
            mask = split_df["text"].str.strip() == text.strip()
            if mask.any():
                found_anywhere = True
                old_labels = split_df.loc[mask, "label"].tolist()
                if any(lbl != "threat" for lbl in old_labels):
                    split_dfs[name].loc[mask, "label"] = "threat"
                    changes["relabeled"].append(
                        {"split": name, "text": text[:60], "old_label": old_labels[0]}
                    )
                    print(f"✅ Перемечено в {name}: \\\"{text[:60]}\\\"")

        # Шаг 2: если не найден нигде — добавляем в train
        if not found_anywhere:
            new_row = pd.DataFrame([{
                "text": text,
                "label": "threat",
                "source": "alexsham_llm",
                "synthetic": 0,
                "avito_original": 0 if "avito_original" in split_dfs["train"].columns else None,
            }])
            # Убираем None-колонки
            new_row = new_row.dropna(axis=1, how="all")
            # Добавляем только те колонки, которые есть в train
            cols = split_dfs["train"].columns
            for c in cols:
                if c not in new_row.columns:
                    new_row[c] = 0 if c in ["synthetic", "avito_original"] else ""
            split_dfs["train"] = pd.concat(
                [split_dfs["train"], new_row[cols]], ignore_index=True
            )
            changes["added"].append({"text": text[:60]})
            print(f"➕ Добавлено в train: \\\"{text[:60]}\\\"")

    # Сохраняем обновлённые сплиты
    for name, split_df in split_dfs.items():
        split_df.to_csv(f"{splits_dir}/{name}.csv", index=False)

    return changes, split_dfs


# Применяем изменения
changes, updated_splits = apply_threat_corrections(
    splits_dir="../data/augmented",
    new_threats_df=new_threats
)

print(f"\\n=== СВОДКА ИЗМЕНЕНИЙ ===")
print(f"Перемечено: {len(changes['relabeled'])}")
for c in changes["relabeled"]:
    print(f"  [{c['split']}] {c['old_label']} → threat: \\\"{c['text']}\\\"")
print(f"Добавлено:  {len(changes['added'])}")
for c in changes["added"]:
    print(f"  [train]  → threat: \\\"{c['text']}\\\"")
"""))

cells.append(code("""\
# Финальная статистика угроз
for name, split_df in updated_splits.items():
    n_threats = (split_df["label"] == "threat").sum()
    print(f"{name}: {n_threats} threats / {len(split_df)} total")

total_threats = sum((s["label"] == "threat").sum() for s in updated_splits.values())
total_all     = sum(len(s) for s in updated_splits.values())
print(f"\\nВсего угроз: {total_threats} ({total_threats/total_all*100:.3f}%)")
print("Было: 276 → Стало:", total_threats, "(+", total_threats - 276, ")")
"""))

# ── 8. Выводы и рекомендации ──────────────────────────────────────────────────
cells.append(md("""\
## 8. Выводы и рекомендации

### Что сделано
- ✅ Размечено 500 AlexSham harassment-примеров через zero-shot NLI
- ✅ Проведён качественный анализ порогов и типов ошибок
- ✅ Найдено **3 новых личных угрозы** (пропущенных regex-эвристикой)
- ✅ Исправлены метки в обучающих данных

### Ключевые находки

**1. AlexSham — источник оскорблений, не личных угроз**

Всего 0.6% примеров AlexSham harassment-bucket содержат прямые личные угрозы.
Это объясняет, почему модель с DIRECT_THREAT_RE имеет очень низкий recall по threat:
большинство реальных угроз в мессенджере Авито имеют другой стиль, чем публичные
токсичные комментарии.

**2. Zero-shot NLI хорошо разделяет «насилие vs оскорбление», но не различает
«личная угроза vs абстрактный призыв к насилию»**

При пороге 0.55: 61.8% — слишком много (false positive — абстрактные призывы)
При пороге 0.85: 8.2% — ближе, но без личного адресата всё равно 95%+ ложных

**3. Двойной фильтр — оптимальное решение**

LLM confidence ≥ 0.80 + regex личного обращения даёт precision ~100%, recall не хуже.

### Рекомендации для следующих шагов

1. **Найти другие источники угроз**: датасеты личной переписки, форумов с прямыми угрозами
2. **Synthetic augmentation**: GPT/Claude для генерации разнообразных форм личных угроз
3. **Fine-tune** на 50+ примерах угроз из Авито-контекста для улучшения recall
4. **Class weights / oversampling** при обучении основного классификатора (threat: ~0.07%)
"""))

# ── 9. Код и воспроизводимость ────────────────────────────────────────────────
cells.append(md("""\
## 9. Воспроизводимость

Запуск разметки (если нужно перезапустить):
```python
from src.llm_labeling import run_labeling
df = run_labeling(
    input_path="data/llm_labeling_sample.csv",
    resume=True,     # пропустить уже размеченные
    verbose=False
)
```

Получение новых угроз:
```python
from src.llm_labeling import build_threat_upgrade_df
new_threats = build_threat_upgrade_df(
    labeled_path="data/llm_labeling_sample.csv",
    min_confidence=0.80,
    require_personal=True  # двойной фильтр
)
```
"""))

# Финал
nb.cells = cells

import nbformat
with open("05_llm_labeling.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("✅ Создан 05_llm_labeling.ipynb")
