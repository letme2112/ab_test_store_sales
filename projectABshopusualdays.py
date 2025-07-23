import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Загрузка данных
df = pd.read_csv("store_sales.csv")

# Приведение форматов
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

# Фильтрация: магазин 1, только обычные дни
store1 = df[(df["Holiday"] == 0) & (df["Store_ID"] == 1)]

# Сводная таблица
summary = (
    store1.groupby("Promo")["Sales"].agg(["mean", "median", "count"]).reset_index()
)
print("\n📋 Сводка по Store 1 (без праздников):\n", summary)

# BoxPlot
plt.figure(figsize=(8, 5))
sns.boxplot(data=store1, x="Promo", y="Sales", palette="Set2")
plt.title("Boxplot продаж по группам Promo (Store 1, без праздников)")
plt.xlabel("Promo (0 = без акции, 1 = с акцией)")
plt.ylabel("Сумма продаж")
plt.grid(True)
plt.tight_layout()
plt.show()

# Time Series
plt.figure(figsize=(12, 6))
sns.lineplot(data=store1, x="Date", y="Sales", hue="Promo", palette="Set1")
plt.title("Динамика продаж во времени по группам Promo (Store 1)")
plt.xlabel("Дата")
plt.ylabel("Сумма продаж")
plt.grid(True)
plt.tight_layout()
plt.show()

# График распределения продаж
plt.figure(figsize=(10, 5))
sns.histplot(data=store1, x="Sales", hue="Promo", bins=40, kde=True, palette="Set1")
plt.title("Распределение продаж (Store 1, без праздников)")
plt.xlabel("Сумма продаж")
plt.ylabel("Количество дней")
plt.grid(True)
plt.tight_layout()
plt.show()

# A/B-тест: Welch's t-test
sales_with_promo = store1[store1["Promo"] == 1]["Sales"]
sales_without_promo = store1[store1["Promo"] == 0]["Sales"]

t_stat, p_value = stats.ttest_ind(
    sales_with_promo, sales_without_promo, equal_var=False
)
mean_diff = sales_with_promo.mean() - sales_without_promo.mean()

# Вывод результатов
print("\n🧪 A/B-тест — Store_ID = 1 (без праздников):")
print(f"  t-статистика: {t_stat:.2f}")
print(f"  p-value: {p_value:.2e}")
print(f"  Разница средних продаж: {mean_diff:.2f}")
print(f"  Размер группы Promo=1: {len(sales_with_promo)}")
print(f"  Размер группы Promo=0: {len(sales_without_promo)}")
