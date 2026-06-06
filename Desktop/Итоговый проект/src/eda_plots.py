"""
Модуль визуализации для EDA датасета классификации сообщений.
Все подписи осей и заголовки — на русском языке.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Optional

# ── Палитра и настройки ───────────────────────────────────────────────────────
PALETTE = {
    "normal":     "#4CAF50",
    "external":   "#2196F3",
    "spam":       "#FF9800",
    "harassment": "#F44336",
    "threat":     "#9C27B0",
}
CLASS_ORDER  = ["normal", "external", "spam", "harassment", "threat"]
CLASS_RU     = {
    "normal":     "Обычное",
    "external":   "Внешний контакт",
    "spam":       "Спам",
    "harassment": "Харассмент",
    "threat":     "Угроза",
}
COLORS = [PALETTE[c] for c in CLASS_ORDER]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def _ru_labels(classes):
    return [CLASS_RU.get(c, c) for c in classes]


def plot_class_distribution(df: pd.DataFrame, ax: plt.Axes, title="Распределение классов"):
    vc = df["label"].value_counts().reindex(CLASS_ORDER).fillna(0)
    bars = ax.bar(_ru_labels(CLASS_ORDER), vc.values, color=COLORS)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Количество (лог. шкала)")
    ax.set_xlabel("Класс")
    ax.tick_params(axis="x", rotation=15)
    for bar, cnt in zip(bars, vc.values):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.4,
                    f"{int(cnt):,}", ha="center", va="bottom", fontsize=8)


def plot_class_pie_minority(df: pd.DataFrame, ax: plt.Axes):
    minority = df[df["label"] != "normal"]["label"].value_counts()
    minority = minority.reindex([c for c in CLASS_ORDER if c != "normal"]).dropna()
    colors = [PALETTE[c] for c in minority.index]
    wedges, texts, autotexts = ax.pie(
        minority.values,
        labels=_ru_labels(minority.index),
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Миноритарные классы\n(без «Обычное»)")


def plot_synthetic_split(df: pd.DataFrame, ax: plt.Axes):
    cross = df.groupby(["label", "source"]).size().unstack(fill_value=0)
    cross = cross.reindex(CLASS_ORDER).fillna(0)
    x = np.arange(len(CLASS_ORDER))
    w = 0.38
    ax.bar(x - w / 2, cross.get("real", 0), w, label="Реальные", color="#4CAF50")
    ax.bar(x + w / 2, cross.get("synthetic", 0), w, label="Синтетические", color="#FF5722")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(_ru_labels(CLASS_ORDER), rotation=15)
    ax.set_title("Реальные vs Синтетические")
    ax.set_ylabel("Количество (лог. шкала)")
    ax.legend(fontsize=9)


def plot_text_length_boxplot(df: pd.DataFrame, ax: plt.Axes,
                              col="text_len", ylabel="Символов", clip=500):
    data = [
        df[df["label"] == c][col].dropna().clip(upper=clip).values
        for c in CLASS_ORDER
    ]
    bp = ax.boxplot(data, patch_artist=True, labels=_ru_labels(CLASS_ORDER))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"Длина текста (клип {clip})")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)


def plot_word_count_violin(df: pd.DataFrame, ax: plt.Axes, clip=50):
    parts = ax.violinplot(
        [df[df["label"] == c]["word_count"].dropna().clip(upper=clip).values
         for c in CLASS_ORDER],
        positions=range(len(CLASS_ORDER)),
        showmedians=True,
        showextrema=False,
    )
    for pc, color in zip(parts["bodies"], COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(_ru_labels(CLASS_ORDER), rotation=15)
    ax.set_title(f"Число слов (клип {clip})")
    ax.set_ylabel("Слов")


def plot_length_density_minority(df: pd.DataFrame, ax: plt.Axes, clip=600):
    minority = [c for c in CLASS_ORDER if c != "normal"]
    for cls in minority:
        vals = df[df["label"] == cls]["text_len"].dropna().clip(upper=clip)
        if len(vals):
            ax.hist(vals, bins=40, alpha=0.5,
                    label=CLASS_RU[cls], color=PALETTE[cls], density=True)
    ax.set_title("Плотность длин (миноритарные классы)")
    ax.set_xlabel("Символов")
    ax.set_ylabel("Плотность")
    ax.legend(fontsize=8)


def full_eda_grid(df: pd.DataFrame, title: str = "EDA датасета",
                  save_path: Optional[str] = None):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    plot_class_distribution(df, axes[0, 0])
    plot_class_pie_minority(df, axes[0, 1])
    plot_synthetic_split(df, axes[0, 2])
    plot_text_length_boxplot(df, axes[1, 0])
    plot_word_count_violin(df, axes[1, 1])
    plot_length_density_minority(df, axes[1, 2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Сохранено: {save_path}")
    return fig


def plot_quality_issues(df_before: pd.DataFrame, df_after: pd.DataFrame,
                        save_path: Optional[str] = None):
    """Сравнение датасета до и после очистки."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Качество данных: до и после очистки", fontsize=14, fontweight="bold")

    # Размеры
    ax = axes[0]
    labels = ["До очистки", "После очистки"]
    sizes  = [len(df_before), len(df_after)]
    bars = ax.bar(labels, sizes, color=["#F44336", "#4CAF50"])
    ax.set_title("Размер датасета")
    ax.set_ylabel("Строк")
    for b, s in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1000,
                f"{s:,}", ha="center", fontsize=10)

    # Нулевые значения
    ax = axes[1]
    null_before = df_before[["text", "label"]].isnull().sum()
    null_after  = df_after[["text", "label"]].isnull().sum() if "label" in df_after.columns else pd.Series([0, 0], index=["text", "label"])
    x = np.arange(2)
    ax.bar(x - 0.2, null_before.values, 0.4, label="До", color="#F44336")
    ax.bar(x + 0.2, null_after.values,  0.4, label="После", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(["Пропуски в тексте", "Пропуски в метке"])
    ax.set_title("Пропущенные значения")
    ax.set_ylabel("Количество")
    ax.legend()

    # Дубликаты
    ax = axes[2]
    dup_before = df_before.duplicated(subset=["text"]).sum()
    dup_after  = df_after.duplicated(subset=["text"]).sum()
    bars = ax.bar(["До очистки", "После очистки"], [dup_before, dup_after],
                  color=["#F44336", "#4CAF50"])
    ax.set_title("Дублирующиеся тексты")
    ax.set_ylabel("Количество")
    for b, v in zip(bars, [dup_before, dup_after]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                str(v), ha="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_train_val_test_split(split_stats: dict, save_path: Optional[str] = None):
    """Визуализация разбивки на train/val/test по классам."""
    splits = ["train", "val", "test"]
    split_ru = {"train": "Обучение", "val": "Валидация", "test": "Тест"}
    colors_split = {"train": "#2196F3", "val": "#FF9800", "test": "#9C27B0"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Разбивка train / val / test", fontsize=14, fontweight="bold")

    # Общий размер по сплитам
    ax = axes[0]
    totals = {s: sum(split_stats[s].values()) for s in splits if s in split_stats}
    ax.bar([split_ru[s] for s in totals],
           list(totals.values()),
           color=[colors_split[s] for s in totals])
    ax.set_title("Общий размер сплитов")
    ax.set_ylabel("Строк")
    for s, v in zip(totals.keys(), totals.values()):
        ax.text(list(split_ru.values()).index(split_ru[s]),
                v + 200, f"{v:,}", ha="center", fontsize=9)

    # Распределение классов по сплитам
    ax = axes[1]
    minority_classes = ["external", "spam", "harassment", "threat"]
    x = np.arange(len(minority_classes))
    w = 0.25
    for i, split in enumerate(splits):
        if split not in split_stats:
            continue
        counts = [split_stats[split].get(c, 0) for c in minority_classes]
        ax.bar(x + (i - 1) * w, counts, w,
               label=split_ru[split], color=colors_split[split])
    ax.set_xticks(x)
    ax.set_xticklabels(_ru_labels(minority_classes), rotation=15)
    ax.set_title("Миноритарные классы по сплитам")
    ax.set_ylabel("Количество")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
