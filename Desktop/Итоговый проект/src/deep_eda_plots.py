"""
Расширенные визуализации для детального EDA аугментированного датасета.
Все подписи на русском языке.
"""

import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional

from src.eda_plots import CLASS_ORDER, CLASS_RU, PALETTE, COLORS

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

SOURCE_COLORS = {
    "real":       "#4CAF50",
    "synthetic":  "#FF5722",
    "alexsham":   "#2196F3",
    "paradetox":  "#9C27B0",
}
SOURCE_RU = {
    "real":      "Авито (реальные)",
    "synthetic": "Синтетические",
    "alexsham":  "AlexSham (ok.ru)",
    "paradetox": "s-nlp/ru_paradetox",
}

STOPWORDS = set(
    "и в на с не я по к за то что он а из это как его до он её они бы уже "
    "есть но у нет же ну так о да ты мне мой нас вас вы вот при всё будет "
    "можно тоже там здесь если для или со при при всей без того со про "
    "этот эта эти был была были чем чего чему когда где куда откуда".split()
)


def _top_words(texts, n=15):
    words = []
    for t in texts.dropna():
        for w in str(t).split():
            w2 = w.lower().strip('.,!?;:\'"()[]{}👍😊📱✅🔴⚡')
            if len(w2) > 2 and w2 not in STOPWORDS:
                words.append(w2)
    return Counter(words).most_common(n)


# ── 1. Общая сводка: размеры и источники ─────────────────────────────────────
def plot_overview(df: pd.DataFrame, save_path: Optional[str] = None):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Сводка аугментированного датасета", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1a. Распределение классов (лог)
    ax = fig.add_subplot(gs[0, 0])
    vc = df["label"].value_counts().reindex(CLASS_ORDER).fillna(0)
    bars = ax.bar([CLASS_RU[c] for c in CLASS_ORDER], vc.values, color=COLORS)
    ax.set_yscale("log")
    ax.set_title("Размер классов (лог. шкала)")
    ax.set_ylabel("Количество")
    ax.tick_params(axis="x", rotation=20)
    for bar, cnt in zip(bars, vc.values):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                    f"{int(cnt):,}", ha="center", va="bottom", fontsize=7)

    # 1b. % каждого класса в датасете
    ax = fig.add_subplot(gs[0, 1])
    pcts = vc / vc.sum() * 100
    colors_pie = COLORS
    wedges, texts, autotexts = ax.pie(
        pcts.values, labels=[CLASS_RU[c] for c in CLASS_ORDER],
        colors=colors_pie, autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
        startangle=90, pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Доли классов")

    # 1c. Источники данных
    ax = fig.add_subplot(gs[0, 2])
    src_vc = df["source"].value_counts()
    src_colors = [SOURCE_COLORS.get(s, "#999") for s in src_vc.index]
    bars2 = ax.barh([SOURCE_RU.get(s, s) for s in src_vc.index], src_vc.values, color=src_colors)
    ax.set_title("Источники строк")
    ax.set_xlabel("Количество")
    for bar, cnt in zip(bars2, src_vc.values):
        ax.text(cnt + 500, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}", va="center", fontsize=8)

    # 1d. Источники по классам (stacked bar)
    ax = fig.add_subplot(gs[1, :2])
    sources = ["real", "synthetic", "alexsham", "paradetox"]
    ct = df.groupby(["label", "source"]).size().unstack(fill_value=0).reindex(CLASS_ORDER)
    x = np.arange(len(CLASS_ORDER))
    bottom = np.zeros(len(CLASS_ORDER))
    for src in sources:
        if src in ct.columns:
            vals = ct[src].values
            ax.bar(x, vals, bottom=bottom,
                   label=SOURCE_RU[src], color=SOURCE_COLORS[src])
            bottom += vals
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_RU[c] for c in CLASS_ORDER], rotation=15)
    ax.set_title("Состав каждого класса по источнику (лог. шкала)")
    ax.set_ylabel("Количество")
    ax.legend(fontsize=8, loc="upper right")

    # 1e. % синтетики по классу
    ax = fig.add_subplot(gs[1, 2])
    synt = df[df["synthetic"] == 1]["label"].value_counts()
    total = df["label"].value_counts()
    synt_pct = (synt / total * 100).reindex(CLASS_ORDER).fillna(0)
    bars3 = ax.bar([CLASS_RU[c] for c in CLASS_ORDER], synt_pct.values,
                   color=["#FF5722" if p > 50 else "#FF9800" if p > 10 else "#4CAF50"
                          for p in synt_pct.values])
    ax.set_title("% синтетических данных в классе")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, p in zip(bars3, synt_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{p:.0f}%", ha="center", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 2. Длины текстов ──────────────────────────────────────────────────────────
def plot_text_lengths(df: pd.DataFrame, save_path: Optional[str] = None):
    df = df.copy()
    df["text_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Анализ длин текстов по классам", fontsize=14, fontweight="bold")

    # a. Boxplot символов
    ax = axes[0, 0]
    data = [df[df["label"] == c]["text_len"].dropna().clip(upper=400).values for c in CLASS_ORDER]
    bp = ax.boxplot(data, patch_artist=True, labels=[CLASS_RU[c] for c in CLASS_ORDER])
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title("Символов (клип 400)")
    ax.set_ylabel("Символов")
    ax.tick_params(axis="x", rotation=20)

    # b. Violin слов
    ax = axes[0, 1]
    parts = ax.violinplot(
        [df[df["label"] == c]["word_count"].dropna().clip(upper=50).values for c in CLASS_ORDER],
        positions=range(len(CLASS_ORDER)), showmedians=True, showextrema=False
    )
    for pc, color in zip(parts["bodies"], COLORS):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels([CLASS_RU[c] for c in CLASS_ORDER], rotation=20)
    ax.set_title("Слов (violin, клип 50)")
    ax.set_ylabel("Слов")

    # c. Гистограмма символов для миноритарных
    ax = axes[0, 2]
    for cls in ["harassment", "threat", "spam", "external"]:
        vals = df[df["label"] == cls]["text_len"].dropna().clip(upper=500)
        ax.hist(vals, bins=40, alpha=0.5, label=CLASS_RU[cls], color=PALETTE[cls], density=True)
    ax.set_title("Плотность длин (миноритарные классы)")
    ax.set_xlabel("Символов")
    ax.set_ylabel("Плотность")
    ax.legend(fontsize=8)

    # d. Биннинг длин
    ax = axes[1, 0]
    df["len_bin"] = pd.cut(df["text_len"],
                           bins=[0, 25, 75, 200, 10000],
                           labels=["<25", "25–75", "75–200", ">200"])
    lb = df.groupby(["label", "len_bin"], observed=False).size().unstack(fill_value=0)
    lb_pct = lb.div(lb.sum(axis=1), axis=0).mul(100).reindex(CLASS_ORDER)
    colors_bin = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    x = np.arange(len(CLASS_ORDER))
    bottom = np.zeros(len(CLASS_ORDER))
    for i, col in enumerate(lb_pct.columns):
        ax.bar(x, lb_pct[col].values, bottom=bottom, color=colors_bin[i], label=str(col))
        bottom += lb_pct[col].fillna(0).values
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_RU[c] for c in CLASS_ORDER], rotation=20)
    ax.set_title("Распределение по длине (% стек)")
    ax.set_ylabel("%")
    ax.legend(title="Символов", fontsize=8)

    # e. Медианная длина по источнику и классу
    ax = axes[1, 1]
    src_order = ["real", "synthetic", "alexsham", "paradetox"]
    x = np.arange(len(CLASS_ORDER))
    w = 0.2
    for i, src in enumerate(src_order):
        sub = df[df["source"] == src]
        medians = [sub[sub["label"] == c]["text_len"].median() for c in CLASS_ORDER]
        medians = [m if not np.isnan(m) else 0 for m in medians]
        ax.bar(x + (i - 1.5) * w, medians, w,
               label=SOURCE_RU.get(src, src), color=SOURCE_COLORS[src])
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_RU[c] for c in CLASS_ORDER], rotation=20)
    ax.set_title("Медианная длина по источнику")
    ax.set_ylabel("Символов (медиана)")
    ax.legend(fontsize=7)

    # f. Outliers (>400 символов) по классу
    ax = axes[1, 2]
    outlier_pct = df.groupby("label").apply(
        lambda x: (x["text_len"] > 400).mean() * 100
    ).reindex(CLASS_ORDER).fillna(0)
    bars = ax.bar([CLASS_RU[c] for c in CLASS_ORDER], outlier_pct.values, color=COLORS)
    ax.set_title("Очень длинные сообщения (>400 симв.), %")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, outlier_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.1f}%", ha="center", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 3. Признаки сигнального типа ─────────────────────────────────────────────
def plot_signal_features(df: pd.DataFrame, save_path: Optional[str] = None):
    df = df.copy()
    df["has_phone"]     = df["text"].str.contains(
        r'\+?7[\s\-\(]?\d{3}|\b\d{10,11}\b', na=False)
    df["has_messenger"] = df["text"].str.contains(
        r'ватсап|вотсап|whatsapp|telegram|телеграм|t\.me/', na=False, flags=re.IGNORECASE)
    df["has_url"]       = df["text"].str.contains(r'https?://', na=False)
    df["has_profanity"] = df["text"].str.contains(
        r'пизд|ёб[её]|хуй|блядь|сук[аи]|ёбан|ебан|бля\b|мразь|тварь',
        na=False, flags=re.IGNORECASE)
    df["has_emoji"]     = df["text"].str.contains(
        r'[\U0001F300-\U0001FFFF]', na=False)
    df["has_caps"]      = df["text"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1) > 0.4)

    features = {
        "Номер телефона":  "has_phone",
        "Мессенджер":      "has_messenger",
        "URL-ссылка":      "has_url",
        "Нецензурная лексика": "has_profanity",
        "Эмодзи":          "has_emoji",
        "Много КАПСЛОКА":  "has_caps",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Сигнальные признаки по классам (% строк)", fontsize=14, fontweight="bold")

    for ax, (title, col) in zip(axes.flatten(), features.items()):
        pcts = df.groupby("label")[col].mean().mul(100).reindex(CLASS_ORDER).fillna(0)
        bars = ax.bar([CLASS_RU[c] for c in CLASS_ORDER], pcts.values, color=COLORS)
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, pcts.values):
            if v > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 4. Топ-слова по классам (горизонтальные бары) ────────────────────────────
def plot_top_words(df: pd.DataFrame, n=12, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 5, figsize=(22, 7))
    fig.suptitle(f"Топ-{n} слов по классам", fontsize=14, fontweight="bold")

    for ax, cls in zip(axes, CLASS_ORDER):
        words_counts = _top_words(df[df["label"] == cls]["text"], n=n)
        if not words_counts:
            ax.set_visible(False)
            continue
        words, counts = zip(*words_counts)
        y = range(len(words))
        ax.barh(list(y), counts, color=PALETTE[cls], alpha=0.85)
        ax.set_yticks(list(y))
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(CLASS_RU[cls])
        ax.set_xlabel("Частота")

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 5. Детальный анализ regex-разметки (threat/harassment) ───────────────────
DIRECT_THREAT_RE = re.compile(
    r"убью\b|убить тебя|убью тебя|пристрелю|застрелю|зарежу|"
    r"сдохни\b|сдохните\b|подохни\b|умри\b|"
    r"тебе\s+пизда\b|пизда\s+тебе\b|тебе\s+конец\b|"
    r"найду\s+тебя|найду\s+вас|приду\s+за\s+тобой|"
    r"займусь\s+тобой|грохну|замочу|прибью|прикончу",
    re.IGNORECASE,
)
TRIGGER_WORDS = [
    "убью", "убить тебя", "пристрелю", "застрелю", "зарежу",
    "сдохни", "сдохните", "подохни", "умри",
    "тебе пизда", "пизда тебе", "тебе конец",
    "найду тебя", "найду вас", "приду за тобой",
    "займусь тобой", "грохну", "замочу", "прибью", "прикончу",
]


def plot_regex_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    alexsham_toxic = df[df["source"] == "alexsham"].copy()
    if len(alexsham_toxic) == 0:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Разметка AlexSham: как regex делит toxic → threat/harassment",
                 fontsize=13, fontweight="bold")

    # a. Итог разбивки
    ax = axes[0]
    counts = alexsham_toxic["label"].value_counts()
    bars = ax.bar([CLASS_RU.get(c, c) for c in counts.index],
                  counts.values,
                  color=[PALETTE.get(c, "#999") for c in counts.index])
    total = counts.sum()
    ax.set_title("Результат regex-разметки\n(источник: AlexSham)")
    ax.set_ylabel("Количество")
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{cnt:,}\n({cnt/total*100:.1f}%)", ha="center", fontsize=9)

    # b. Частота срабатывания каждого слова-триггера
    ax = axes[1]
    trigger_counts = {}
    for tw in TRIGGER_WORDS:
        pat = re.compile(re.escape(tw), re.IGNORECASE)
        n = alexsham_toxic["text"].str.contains(pat, regex=True, na=False).sum()
        if n > 0:
            trigger_counts[tw] = n
    if trigger_counts:
        sorted_tc = sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)
        words, cnts = zip(*sorted_tc)
        ax.barh(list(words), list(cnts), color="#F44336", alpha=0.8)
        ax.invert_yaxis()
        ax.set_title("Слова-триггеры для класса threat\n(сколько раз сработали)")
        ax.set_xlabel("Количество срабатываний")
        for i, (w, c) in enumerate(zip(words, cnts)):
            ax.text(c + 0.5, i, str(c), va="center", fontsize=8)

    # c. Длина текстов threat vs harassment (из AlexSham)
    ax = axes[2]
    for cls, color in [("threat", PALETTE["threat"]), ("harassment", PALETTE["harassment"])]:
        sub = alexsham_toxic[alexsham_toxic["label"] == cls]["text"].str.len().clip(upper=300)
        ax.hist(sub, bins=30, alpha=0.6, label=CLASS_RU[cls], color=color, density=True)
    ax.set_title("Длина текстов: threat vs harassment\n(только AlexSham)")
    ax.set_xlabel("Символов")
    ax.set_ylabel("Плотность")
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 6. Сравнение распределений: Авито-реальные vs внешние ──────────────────
def plot_domain_comparison(df: pd.DataFrame, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Сравнение доменов: Авито-оригинал vs Внешние источники",
                 fontsize=13, fontweight="bold")

    df = df.copy()
    df["text_len"] = df["text"].str.len()

    # a. Длина harassment: real vs alexsham vs paradetox
    ax = axes[0]
    for src, color in [("real", SOURCE_COLORS["real"]),
                       ("alexsham", SOURCE_COLORS["alexsham"]),
                       ("paradetox", SOURCE_COLORS["paradetox"])]:
        sub = df[(df["label"] == "harassment") & (df["source"] == src)]["text_len"].clip(upper=400)
        if len(sub):
            ax.hist(sub, bins=35, alpha=0.5, density=True,
                    label=SOURCE_RU[src], color=color)
    ax.set_title("Harassment: длина по источнику")
    ax.set_xlabel("Символов")
    ax.set_ylabel("Плотность")
    ax.legend(fontsize=8)

    # b. Длина threat
    ax = axes[1]
    for src, color in [("real", SOURCE_COLORS["real"]),
                       ("alexsham", SOURCE_COLORS["alexsham"]),
                       ("synthetic", SOURCE_COLORS["synthetic"])]:
        sub = df[(df["label"] == "threat") & (df["source"] == src)]["text_len"].clip(upper=300)
        if len(sub):
            ax.hist(sub, bins=25, alpha=0.5, density=True,
                    label=SOURCE_RU[src], color=color)
    ax.set_title("Threat: длина по источнику")
    ax.set_xlabel("Символов")
    ax.set_ylabel("Плотность")
    ax.legend(fontsize=8)

    # c. Боксплоты длин harassment по источнику
    ax = axes[2]
    sources_present = [s for s in ["real", "synthetic", "alexsham", "paradetox"]
                       if len(df[(df["label"] == "harassment") & (df["source"] == s)]) > 0]
    data_bp = [df[(df["label"] == "harassment") & (df["source"] == s)]["text_len"]
                .clip(upper=400).values for s in sources_present]
    bp = ax.boxplot(data_bp, patch_artist=True,
                    labels=[SOURCE_RU.get(s, s) for s in sources_present])
    for patch, src in zip(bp["boxes"], sources_present):
        patch.set_facecolor(SOURCE_COLORS[src])
        patch.set_alpha(0.7)
    ax.set_title("Harassment: боксплот длин по источнику")
    ax.set_ylabel("Символов (клип 400)")
    ax.tick_params(axis="x", rotation=15)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ── 7. Риски и проблемы ───────────────────────────────────────────────────────
def plot_risks(df: pd.DataFrame, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Риски датасета: дисбаланс, синтетика, малые классы",
                 fontsize=13, fontweight="bold")

    # a. Imbalance ratio (относительно самого частого класса)
    ax = axes[0]
    vc = df["label"].value_counts().reindex(CLASS_ORDER)
    max_cls = vc.max()
    ratios = (max_cls / vc).round(0)
    bars = ax.bar([CLASS_RU[c] for c in CLASS_ORDER], ratios.values, color=COLORS)
    ax.set_title("Коэффициент дисбаланса\n(normal / класс)")
    ax.set_ylabel("Раз")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, ratios.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"×{int(v):,}", ha="center", fontsize=8)

    # b. % синтетики / внешних источников — «ненадёжных» данных
    ax = axes[1]
    unreliable = df[df["source"].isin(["synthetic", "alexsham", "paradetox"])]
    unreliable_pct = (unreliable.groupby("label").size() /
                      df.groupby("label").size() * 100).reindex(CLASS_ORDER).fillna(0)
    colors_risk = ["#4CAF50" if p < 10 else "#FF9800" if p < 50 else "#F44336"
                   for p in unreliable_pct.values]
    bars2 = ax.bar([CLASS_RU[c] for c in CLASS_ORDER], unreliable_pct.values, color=colors_risk)
    ax.axhline(50, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="50%")
    ax.axhline(90, color="darkred", linestyle=":", linewidth=0.8, alpha=0.6, label="90%")
    ax.set_title("% ненадёжных строк\n(синтетика + внешние)")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    for bar, v in zip(bars2, unreliable_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.0f}%", ha="center", fontsize=8)

    # c. Размер реального val (только Авито) по классу
    ax = axes[2]
    val_df = df[df["label"].isin(CLASS_ORDER)]
    real_counts = df[df["source"] == "real"]["label"].value_counts().reindex(CLASS_ORDER).fillna(0)
    val_counts  = (real_counts * 0.10).round(0)
    bars3 = ax.bar([CLASS_RU[c] for c in CLASS_ORDER], val_counts.values, color=COLORS)
    ax.set_yscale("log")
    ax.set_title("Реальных примеров в val/test\n(≈10% реальных Авито)")
    ax.set_ylabel("Количество (лог)")
    ax.tick_params(axis="x", rotation=20)
    for bar, v in zip(bars3, val_counts.values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.5,
                    int(v), ha="center", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
