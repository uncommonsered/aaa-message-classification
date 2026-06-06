"""
EDA: final_labeled_data_v1.csv
Датасет: сообщения мессенджера Авито с метками токсичности
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "final_labeled_data_v1.csv"

# ─── Загрузка ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ─── 1. Базовая информация ───────────────────────────────────────────────────
print("=" * 50)
print("1. BASIC INFO")
print("=" * 50)
print(df.dtypes)
print(f"\nNull counts:\n{df.isnull().sum()}")
print(f"\nDuplicates (by text): {df.duplicated(subset=['text']).sum()}")
print(f"Duplicates (text+sentiment): {df.duplicated(subset=['text','sentiment']).sum()}")

# ─── 2. Распределение классов ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. CLASS DISTRIBUTION")
print("=" * 50)
vc = df["sentiment"].value_counts()
vcp = df["sentiment"].value_counts(normalize=True) * 100
dist = pd.DataFrame({"count": vc, "pct": vcp.round(2)})
print(dist.to_string())

# ─── 3. Синтетические vs реальные данные ────────────────────────────────────
print("\n" + "=" * 50)
print("3. SYNTHETIC vs REAL by class")
print("=" * 50)
cross = df.groupby(["sentiment", "synthetic"]).size().unstack(fill_value=0)
cross.columns = ["real", "synthetic"]
cross["synthetic_pct"] = (cross["synthetic"] / (cross["real"] + cross["synthetic"]) * 100).round(1)
print(cross.to_string())

# ─── 4. Длина текста по классам ──────────────────────────────────────────────
df["text_len"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()

print("\n" + "=" * 50)
print("4. TEXT LENGTH BY CLASS (chars)")
print("=" * 50)
print(df.groupby("sentiment")["text_len"].describe().round(1).to_string())

print("\n" + "=" * 50)
print("5. WORD COUNT BY CLASS")
print("=" * 50)
print(df.groupby("sentiment")["word_count"].describe().round(1).to_string())

# ─── 5. Примеры каждого класса ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("6. EXAMPLES PER CLASS (non-synthetic)")
print("=" * 50)
for cls in ["harassment", "threat", "spam", "external", "normal"]:
    subset = df[(df["sentiment"] == cls) & (df["synthetic"] == 0)]["text"].dropna()
    print(f"\n--- {cls} ({len(df[df['sentiment']==cls])} total) ---")
    for ex in subset.sample(min(3, len(subset)), random_state=42):
        print(f"  {str(ex)[:130]}")

# ─── Визуализации ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("EDA: Message Classification Dataset", fontsize=15, fontweight="bold")

CLASS_ORDER = ["normal", "external", "spam", "harassment", "threat"]
COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]

# --- Plot 1: Bar chart распределения классов (лог шкала) ---
ax = axes[0, 0]
counts = [vc.get(c, 0) for c in CLASS_ORDER]
bars = ax.bar(CLASS_ORDER, counts, color=COLORS)
ax.set_yscale("log")
ax.set_title("Class Distribution (log scale)")
ax.set_ylabel("Count (log)")
ax.tick_params(axis="x", rotation=20)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
            f"{cnt:,}", ha="center", va="bottom", fontsize=8)

# --- Plot 2: Пирог без normal (для видимости минорных классов) ---
ax = axes[0, 1]
minor = df[df["sentiment"] != "normal"]["sentiment"].value_counts()
ax.pie(minor.values, labels=minor.index, autopct="%1.1f%%",
       colors=["#2196F3", "#FF9800", "#F44336", "#9C27B0"], startangle=140)
ax.set_title("Minority Classes\n(excluding 'normal')")

# --- Plot 3: Synthetic split ---
ax = axes[0, 2]
cross_plot = cross[["real", "synthetic"]].reindex(CLASS_ORDER).fillna(0)
x = np.arange(len(CLASS_ORDER))
w = 0.4
ax.bar(x - w/2, cross_plot["real"], w, label="Real", color="#4CAF50")
ax.bar(x + w/2, cross_plot["synthetic"], w, label="Synthetic", color="#FF5722")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_ORDER, rotation=20)
ax.set_title("Real vs Synthetic (log scale)")
ax.set_ylabel("Count (log)")
ax.legend()

# --- Plot 4: Boxplot длин текстов ---
ax = axes[1, 0]
data_bp = [df[df["sentiment"] == c]["text_len"].dropna().clip(upper=500).values for c in CLASS_ORDER]
bp = ax.boxplot(data_bp, patch_artist=True, labels=CLASS_ORDER)
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Text Length Distribution (chars, clipped at 500)")
ax.set_ylabel("Chars")
ax.tick_params(axis="x", rotation=20)

# --- Plot 5: Boxplot числа слов ---
ax = axes[1, 1]
data_wc = [df[df["sentiment"] == c]["word_count"].dropna().clip(upper=60).values for c in CLASS_ORDER]
bp2 = ax.boxplot(data_wc, patch_artist=True, labels=CLASS_ORDER)
for patch, color in zip(bp2["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Word Count Distribution (clipped at 60)")
ax.set_ylabel("Words")
ax.tick_params(axis="x", rotation=20)

# --- Plot 6: Гистограмма длин по non-normal классам ---
ax = axes[1, 2]
for cls, color in zip(["harassment", "threat", "spam", "external"], COLORS[1:]):
    subset = df[(df["sentiment"] == cls)]["text_len"].dropna().clip(upper=600)
    ax.hist(subset, bins=40, alpha=0.5, label=cls, color=color, density=True)
ax.set_title("Text Length Density (minority classes)")
ax.set_xlabel("Chars")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig("eda_report.png", dpi=120, bbox_inches="tight")
print("\n✓ Saved: eda_report.png")
plt.show()

# ─── Итоговые наблюдения ─────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("SUMMARY OF FINDINGS")
print("=" * 50)
print(f"""
1. СИЛЬНЫЙ ДИСБАЛАНС КЛАССОВ
   - normal: 97.07% (356 954 сообщений)
   - external: 2.23% (8 190) — сообщения о внешних контактах (WhatsApp, телефон)
   - spam: 0.60% (2 220)
   - harassment: 0.08% (291)
   - threat: 0.02% (81)
   → Нужны oversampling / class weights / threshold tuning

2. СИНТЕТИЧЕСКИЕ ДАННЫЕ
   - harassment: 70% синтетика (204/291)
   - threat: 95% синтетика (77/81)
   - spam: 90% синтетика (2000/2220)
   - external: 47% синтетика
   → Качество синтетики критично для minority-классов

3. ДЛИНА ТЕКСТОВ
   - external и spam значительно длиннее (медиана ~76–97 символов)
   - normal, threat, harassment — короче (медиана 35–47 символов)
   - Длина может быть полезным признаком

4. ПРОПУСКИ
   - 122 пропуска в text, 8 в sentiment → нужна чистка перед обучением

5. ДУБЛИКАТЫ
   - 309 дубликатов по тексту → удалить перед обучением
""")
