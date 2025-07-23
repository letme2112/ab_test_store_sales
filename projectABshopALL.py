import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Загрузка и подготовка данных
df = pd.read_csv("store_sales.csv")

df.rename(
    columns={
        "date": "Date",
        "store": "Store_ID",
        "sales": "Sales",
        "promo": "Promo",
        "holiday": "Holiday",
    },
    inplace=True,
)

df["Date"] = pd.to_datetime(df["Date"])

# Сводная таблица по Promo
summary = df.groupby("Promo")["Sales"].agg(["mean", "median", "count"]).reset_index()
print("\n📊 Сводка по группам Promo:\n", summary)

# Гистограмма распределения продаж
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="Sales", hue="Promo", bins=50, kde=True, palette="Set1")
plt.title("Распределение продаж по группам Promo")
plt.xlabel("Сумма продаж")
plt.ylabel("Количество дней")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot: продажи по группам Promo
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Promo", y="Sales", palette="Set2")
plt.title("Boxplot продаж по группам Promo (все магазины и дни)")
plt.xlabel("Promo (0 = без акции, 1 = с акцией)")
plt.ylabel("Сумма продаж")
plt.grid(True)
plt.tight_layout()
plt.show()

# Time series
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Date", y="Sales", hue="Promo", palette="Set1")
plt.title("Динамика продаж по группам Promo (все магазины и дни)")
plt.xlabel("Дата")
plt.ylabel("Сумма продаж")
plt.grid(True)
plt.tight_layout()
plt.show()

# A/B-тест: сравнение продаж
sales_with_promo = df[df["Promo"] == 1]["Sales"]
sales_without_promo = df[df["Promo"] == 0]["Sales"]

t_stat, p_value = stats.ttest_ind(
    sales_with_promo, sales_without_promo, equal_var=False
)
mean_diff = sales_with_promo.mean() - sales_without_promo.mean()

print(f"\n🧪 A/B-тест (t-test):")
print(f"  t-статистика: {t_stat:.2f}")
print(f"  p-value: {p_value:.3e}")
print(f"  Разница средних: {mean_diff:.2f}")
