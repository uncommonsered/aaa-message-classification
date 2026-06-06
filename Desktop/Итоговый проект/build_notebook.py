"""Генерирует 01_eda_and_data_prep.ipynb программно."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3.10.0"}
})

cells = []

def md(source):
    return nbf.v4.new_markdown_cell(source)

def code(source):
    return nbf.v4.new_code_cell(source)

# ─── Заголовок ───────────────────────────────────────────────────────────────
cells.append(md("""# Классификация сообщений мессенджера Авито
## EDA и подготовка датасета

**Цель проекта**: автоматически определять, является ли сообщение нормальным, спамом, харассментом или угрозой, и ограничивать отправку нежелательного контента.

**Исходные данные**:
- `messages_project_aaa.csv` — 492 800 сырых сообщений с `message_id`, `text`, `item_title`
- `final_labeled_data_v1.csv` — 367 744 размеченных строк (включая синтетику)

**Структура ноутбука**:
1. Загрузка и первичный осмотр
2. Диагностика качества датасета
3. EDA: распределения, длины, примеры
4. Подготовка тренировочного датасета
5. Train / Val / Test разбивка
6. Итоги и выводы
"""))

# ─── 0. Импорты ──────────────────────────────────────────────────────────────
cells.append(md("## 0. Импорты и настройки"))
cells.append(code("""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # ноутбук запускается из папки notebooks/
                                             # или из корня проекта

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# Локальные модули
from src.data_prep import (
    load_raw, load_labeled, merge_with_labels,
    clean, add_text_features, make_train_ready,
    split, save_splits, split_stats, quality_report,
    NORMAL_CAP,
)
from src.eda_plots import (
    full_eda_grid, plot_quality_issues,
    plot_train_val_test_split, CLASS_RU, CLASS_ORDER,
)

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.float_format", "{:.1f}".format)
print("✓ Импорты успешны")
"""))

# ─── 1. Загрузка ──────────────────────────────────────────────────────────────
cells.append(md("## 1. Загрузка данных"))
cells.append(code("""\
RAW_PATH     = "messages_project_aaa.csv"
LABELED_PATH = "final_labeled_data_v1.csv"

raw     = load_raw(RAW_PATH)
labeled = load_labeled(LABELED_PATH)

print("=== Сырые данные (messages_project_aaa) ===")
display(raw.head(3))
print(f"\\nРазмер: {raw.shape[0]:,} строк × {raw.shape[1]} колонки")
print(f"Уникальных текстов: {raw['text'].nunique():,}")
print(f"Дублей по тексту: {raw.duplicated(subset=['text']).sum():,}")
"""))

cells.append(code("""\
print("=== Размеченные данные (final_labeled_data_v1) ===")
display(labeled.head(3))
print(f"\\nРазмер: {labeled.shape[0]:,} строк × {labeled.shape[1]} колонки")
print(f"\\nРаспределение классов:")
display(labeled["label"].value_counts().rename("count").to_frame())
"""))

# ─── 2. Диагностика качества ─────────────────────────────────────────────────
cells.append(md("""## 2. Диагностика качества датасета

### Ключевые проблемы:

| Проблема | Описание |
|---|---|
| **Критический дисбаланс** | 97% сообщений — класс «normal» |
| **Мало реальных примеров** | Угрозы: 4 реальных (95% синтетика!), Харассмент: 87 реальных |
| **Пропуски** | 122 пустых текста, 8 без метки |
| **Дубликаты** | 131 200 дублей в raw-файле |
| **Синтетика** | 70–95% миноритарных классов — сгенерированные данные |
"""))

cells.append(code("""\
print("=== Пропущенные значения в labeled ===")
print(labeled[["text", "label"]].isnull().sum())

print("\\n=== Дубликаты ===")
print(f"Дублей по тексту: {labeled.duplicated(subset=['text']).sum():,}")
print(f"Дублей по (текст+метка): {labeled.duplicated(subset=['text','label']).sum():,}")

print("\\n=== Реальные vs Синтетические по классу ===")
cross = labeled.groupby(["label", "source"]).size().unstack(fill_value=0)
cross.columns = ["Реальные", "Синтетические"]
cross["% синтетики"] = (
    cross["Синтетические"] / (cross["Реальные"] + cross["Синтетические"]) * 100
).round(1)
display(cross.reindex(CLASS_ORDER).fillna(0))
"""))

# ─── 3. EDA ──────────────────────────────────────────────────────────────────
cells.append(md("## 3. EDA: исходный датасет"))
cells.append(code("""\
# Подготовим датасет с признаками для EDA
df_eda = add_text_features(labeled.dropna(subset=["text", "label"]))

# Сводная таблица по классам
print("=== Описательная статистика длины текста по классам ===")
display(df_eda.groupby("label")[["text_len", "word_count"]].describe().round(1))
"""))

cells.append(code("""\
fig = full_eda_grid(df_eda, title="EDA: исходный датасет (final_labeled_data_v1)",
                    save_path="eda_report_initial.png")
plt.show()
"""))

cells.append(code("""\
print("=== Примеры по классам (только реальные) ===")
for cls in CLASS_ORDER:
    subset = df_eda[(df_eda["label"] == cls) & (df_eda["source"] == "real")]["text"].dropna()
    n_real = len(subset)
    print(f"\\n{'─'*60}")
    print(f"Класс: {CLASS_RU[cls]} | Реальных: {n_real}")
    print(f"{'─'*60}")
    for txt in subset.sample(min(3, n_real), random_state=42):
        print(f"  {str(txt)[:120]}")
"""))

# ─── 4. Подготовка датасета ───────────────────────────────────────────────────
cells.append(md("""## 4. Подготовка тренировочного датасета

### Стратегия:
1. **Объединяем** raw (с `message_id`) + labeled (с метками) — через текст
2. **Очищаем**: удаляем NaN, дубли, строки < 2 символов
3. **Ограничиваем** класс `normal` сверху (не нарушая реальные пропорции) — иначе модель выучит тривиальное «всё нормально»
4. **Синтетические** примеры идут **только в train** — val/test содержат исключительно реальные данные
"""))

cells.append(code("""\
# Объединение
df_merged = merge_with_labels(raw, labeled)
print(f"После merge: {len(df_merged):,} строк")
print(f"Наличие message_id: {df_merged['message_id'].notna().sum():,} реальных строк")
"""))

cells.append(code("""\
# Очистка
df_clean = clean(df_merged)
df_clean = add_text_features(df_clean)

print("\\n=== После очистки ===")
display(quality_report(df_clean))
"""))

cells.append(code("""\
fig_q = plot_quality_issues(
    df_merged.rename(columns={"label": "label"}),
    df_clean,
    save_path="eda_quality.png"
)
plt.show()
"""))

cells.append(code("""\
# Балансировка
df_train_ready = make_train_ready(df_clean, normal_cap=NORMAL_CAP)

print(f"\\nДатасет после балансировки: {len(df_train_ready):,} строк")
print("\\n=== Распределение классов ===")
display(df_train_ready["label"].value_counts().rename("count"))

fig2 = full_eda_grid(df_train_ready,
                      title=f"EDA: сбалансированный датасет (normal cap={NORMAL_CAP:,})",
                      save_path="eda_report_balanced.png")
plt.show()
"""))

# ─── 5. Train / Val / Test ───────────────────────────────────────────────────
cells.append(md("""## 5. Train / Val / Test разбивка

**Правило**: val и test — **только реальные** примеры.
Синтетические данные остаются в train.
"""))

cells.append(code("""\
train, val, test = split(df_train_ready, val_size=0.10, test_size=0.10)

stats = split_stats(train, val, test)

print("=== Train ===")
display(train["label"].value_counts().rename("count"))
print(f"  из них синтетических: {(train['source']=='synthetic').sum():,}")

print("\\n=== Val ===")
display(val["label"].value_counts().rename("count"))

print("\\n=== Test ===")
display(test["label"].value_counts().rename("count"))
"""))

cells.append(code("""\
fig3 = plot_train_val_test_split(stats, save_path="eda_splits.png")
plt.show()
"""))

cells.append(code("""\
# Сохраняем на диск
save_splits(train, val, test, out_dir="data/processed")
print("\\nКолонки в train.csv:", train.columns.tolist())
"""))

# ─── 6. Итоги ────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Итоги и выводы

### Результат подготовки данных

| | Исходный | После очистки | Train | Val | Test |
|---|---|---|---|---|---|
| Строк | 367 744 | ~367 000 | см. ниже | см. ниже | см. ниже |
| % синтетики | 1.7% | 1.7% | содержит | 0% | 0% |
"""))

cells.append(code("""\
summary = {
    "Исходный датасет":   len(labeled),
    "После очистки":      len(df_clean),
    "Train (с синтетикой)": len(train),
    "Val (только real)":  len(val),
    "Test (только real)": len(test),
}
print("=== Сводка размеров ===")
for k, v in summary.items():
    print(f"  {k:<28}: {v:>8,}")

print("\\n=== Финальное качество: таблица по классам ===")
display(quality_report(df_clean))
"""))

cells.append(md("""### Выводы

1. **Дисбаланс — главная проблема**: 97% сообщений — класс «normal». При обучении обязательно использовать `class_weight='balanced'` или oversampling (SMOTE на эмбеддингах).

2. **Синтетика**: для `threat` — 95% синтетики, для `harassment` — 70%. Без неё датасет практически непригоден для обучения миноритарных классов. Качество синтетических примеров нужно контролировать.

3. **Разбивка val/test — только real**: это гарантирует честную оценку модели на реальных данных (без утечки синтетики в метрики).

4. **Рекомендуемые метрики**: Precision ≥ 0.95 по классу `normal` (бизнес-требование), macro-F1 для остальных классов. Матрица ошибок с акцентом на минимизацию FN для `threat` и `harassment`.

5. **Следующий шаг**: обучение классификатора на базе предобученного многоязычного encoder'а (например, `cointegrated/rubert-tiny2` или `DeepPavlov/rubert-base-cased`) с fine-tuning на подготовленном train.csv.
"""))

# ─── Финал ──────────────────────────────────────────────────────────────────
nb.cells = cells
nbf.write(nb, "/Users/bulletqueen/Desktop/Итоговый проект/01_eda_and_data_prep.ipynb")
print("Ноутбук создан: 01_eda_and_data_prep.ipynb")
