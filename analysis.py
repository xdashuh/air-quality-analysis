"""
Case Study: Air Quality Analysis (India, 2015-2020)

Run:  python analysis.py
Needs: data/city_day.csv  (Kaggle: "Air Quality Data in India" by Rohan Rao)

Cleans the data, analyses it, runs statistical tests and a prediction model,
saves charts to figures/ and every number used in the report to results.json.
"""
import os
import json
import warnings

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

sns.set_theme(style="whitegrid")

DATA = "data/city_day.csv"
if not os.path.exists(DATA):
    raise SystemExit(
        "\nMissing data/city_day.csv\n"
        "Download it from Kaggle: 'Air Quality Data in India' (Rohan Rao), file city_day.csv,\n"
        "and place it inside the data/ folder. See README.md.\n")

POLL = ["PM25", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
BUCKETS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
SEASONS = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
res = {"figures": []}


def fig_done(name, caption):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=150)
    plt.close("all")
    res["figures"].append({"file": f"{name}.png", "caption": caption})


def season_of(m):
    return ("Winter" if m in (12, 1, 2) else "Summer" if m in (3, 4, 5)
            else "Monsoon" if m in (6, 7, 8, 9) else "Post-monsoon")


def bucket_of(a):
    if pd.isna(a):
        return np.nan
    for limit, name in [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor")]:
        if a <= limit:
            return name
    return "Severe"


# =====================================================================  1. LOAD
raw = pd.read_csv(DATA)
raw = raw.rename(columns={"PM2.5": "PM25"})
res["raw_rows"], res["raw_cols"] = int(raw.shape[0]), int(raw.shape[1])
res["columns"] = raw.columns.tolist()
res["raw_dtypes"] = {k: str(v) for k, v in raw.dtypes.items()}
res["raw_missing_pct"] = (raw.isna().mean() * 100).round(1).to_dict()
res["raw_duplicates"] = int(raw.duplicated(subset=["City", "Date"]).sum())

# =====================================================================  2. CLEAN
df = raw.copy()

# incorrect data types -> proper datetime / numeric
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
res["bad_dates"] = int(df["Date"].isna().sum())
df = df.dropna(subset=["Date"])
for c in POLL + ["AQI"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# inconsistent values -> tidy text labels
df["City"] = df["City"].astype(str).str.strip()
df["AQI_Bucket"] = df["AQI_Bucket"].astype("string").str.strip().str.title()
df["AQI_Bucket"] = df["AQI_Bucket"].replace({"Very poor": "Very Poor"})

# duplicates
df = df.drop_duplicates(subset=["City", "Date"]).sort_values(["City", "Date"]).reset_index(drop=True)

# impossible values (negative concentrations) -> missing
neg = {c: int((df[c] < 0).sum()) for c in POLL + ["AQI"]}
res["negative_values"] = {k: v for k, v in neg.items() if v}
for c in POLL + ["AQI"]:
    df.loc[df[c] < 0, c] = np.nan

# missing values: fill only SHORT gaps (<= 7 days) within each city by linear interpolation
before = df[["PM25", "PM10", "NO2", "AQI"]].isna().sum()
for c in POLL:
    df[c] = df.groupby("City")[c].transform(lambda s: s.interpolate(limit=7, limit_area="inside"))
after = df[["PM25", "PM10", "NO2", "AQI"]].isna().sum()
res["filled_by_interpolation"] = (before - after).astype(int).to_dict()
res["missing_after_pct"] = (df[POLL + ["AQI"]].isna().mean() * 100).round(1).to_dict()

# rebuild missing AQI buckets from the AQI value (official CPCB bands)
res["bucket_filled"] = int((df["AQI_Bucket"].isna() & df["AQI"].notna()).sum())
df["AQI_Bucket"] = df["AQI_Bucket"].fillna(df["AQI"].map(bucket_of))

# derived columns
df["Year"], df["Month"] = df["Date"].dt.year, df["Date"].dt.month
df["Season"] = df["Month"].map(season_of)

res["clean_rows"] = int(len(df))
res["n_cities"] = int(df["City"].nunique())
res["date_min"], res["date_max"] = str(df["Date"].min().date()), str(df["Date"].max().date())
df.to_csv("data/city_day_clean.csv", index=False)

# =====================================================================  3. ANALYSIS
# --- 3.1 descriptive statistics
cols = ["PM25", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
res["describe"] = df[cols].agg(["mean", "median", "std"]).round(2).to_dict()
res["np_pm25_mean"] = round(float(np.nanmean(df["PM25"])), 2)
res["np_pm25_std"] = round(float(np.nanstd(df["PM25"], ddof=1)), 2)

# --- 3.2 city comparison (cities with at least a year of AQI records)
cnt = df.dropna(subset=["AQI"]).groupby("City")["AQI"].agg(["mean", "count"])
city = cnt[cnt["count"] >= 365].sort_values("mean", ascending=False)
res["city_table"] = [{"city": c, "aqi": round(r["mean"], 1), "days": int(r["count"])}
                     for c, r in city.head(10).iterrows()]
res["worst_city"], res["worst_city_aqi"] = city.index[0], round(city["mean"].iloc[0], 1)
res["best_city"], res["best_city_aqi"] = city.index[-1], round(city["mean"].iloc[-1], 1)
res["n_cities_ranked"] = int(len(city))

# --- 3.3 days in each AQI category
b = df["AQI_Bucket"].value_counts().reindex(BUCKETS).fillna(0).astype(int)
res["bucket_counts"] = b.to_dict()
res["bucket_pct"] = (b / b.sum() * 100).round(1).to_dict()
poor_plus = df[df["AQI_Bucket"].isin(["Poor", "Very Poor", "Severe"])]
res["poor_plus_pct"] = round(100 * len(poor_plus) / b.sum(), 1)
share = (df.dropna(subset=["AQI_Bucket"]).assign(pp=lambda d: d["AQI_Bucket"].isin(["Poor", "Very Poor", "Severe"]))
         .groupby("City").agg(pp=("pp", "mean"), n=("pp", "size")))
share = share[share["n"] >= 365].sort_values("pp", ascending=False)
res["most_poor_city"], res["most_poor_city_pct"] = share.index[0], round(100 * share["pp"].iloc[0], 1)

# --- 3.4 seasonal pattern
sea = df.groupby("Season")["PM25"].agg(["mean", "median", "count"]).reindex(SEASONS)
res["season_table"] = [{"season": s, "mean": round(r["mean"], 1), "median": round(r["median"], 1)} for s, r in sea.iterrows()]
res["worst_season"], res["best_season"] = sea["mean"].idxmax(), sea["mean"].idxmin()
res["season_ratio"] = round(sea["mean"].max() / sea["mean"].min(), 2)
monthly = df.groupby("Month")["PM25"].mean()
res["worst_month"], res["best_month"] = int(monthly.idxmax()), int(monthly.idxmin())

# --- 3.5 yearly trend (city-year means, so a change in which cities report does not distort it)
cy = df.dropna(subset=["PM25"]).groupby(["City", "Year"])["PM25"].agg(["mean", "count"]).reset_index()
cy = cy[cy["count"] >= 100]
yearly = cy.groupby("Year")["mean"].agg(["mean", "count"])
res["yearly_table"] = [{"year": int(y), "pm25": round(r["mean"], 1), "cities": int(r["count"])} for y, r in yearly.iterrows()]

# --- 3.6 correlation
corr = df[POLL + ["AQI"]].corr()
res["corr_aqi"] = corr["AQI"].drop("AQI").round(2).sort_values(ascending=False).to_dict()
res["top_corr_with_aqi"] = list(res["corr_aqi"].items())[:3]
pair = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(ascending=False)
res["strongest_pair"] = [pair.index[0][0], pair.index[0][1], round(float(pair.iloc[0]), 2)]
pm_pair = df[["PM25", "PM10"]].dropna()
rho, p_rho = stats.spearmanr(pm_pair["PM25"], pm_pair["PM10"])
res["pm25_pm10_spearman"], res["pm25_pm10_p"] = round(float(rho), 3), float(p_rho)
res["pm25_pm10_ratio"] = round(float((pm_pair["PM25"] / pm_pair["PM10"]).median()), 2)

# --- 3.7 statistical test: do seasons differ?  (Kruskal-Wallis, non-parametric)
groups = [df.loc[df["Season"] == s, "PM25"].dropna() for s in SEASONS]
H, p_kw = stats.kruskal(*groups)
res["kruskal_H"], res["kruskal_p"] = round(float(H), 1), float(p_kw)

# --- 3.8 COVID-19 lockdown effect: 25 Mar - 30 Apr, 2020 vs the same dates in 2019
def window(y):
    d = df[df["Year"] == y]
    d = d[((d["Month"] == 3) & (d["Date"].dt.day >= 25)) | (d["Month"] == 4)]
    return d.groupby("City")["PM25"].agg(["mean", "count"])

w19, w20 = window(2019), window(2020)
j = w19.join(w20, lsuffix="_19", rsuffix="_20", how="inner")
j = j[(j["count_19"] >= 10) & (j["count_20"] >= 10)]
res["lockdown_n_cities"] = int(len(j))
if len(j) >= 5:
    res["lockdown_mean_2019"], res["lockdown_mean_2020"] = round(j["mean_19"].mean(), 1), round(j["mean_20"].mean(), 1)
    res["lockdown_change_pct"] = round(100 * (j["mean_20"].mean() / j["mean_19"].mean() - 1), 1)
    res["lockdown_cities_improved"] = int((j["mean_20"] < j["mean_19"]).sum())
    w_stat, p_w = stats.wilcoxon(j["mean_19"], j["mean_20"])
    res["lockdown_p"] = float(p_w)
    lock = j.sort_values("mean_19", ascending=False).head(8)

# --- 3.9 prediction model: tomorrow's PM2.5 from today's readings
m = df.copy()
m["target"] = m.groupby("City")["PM25"].shift(-1)
m["gap"] = (m.groupby("City")["Date"].shift(-1) - m["Date"]).dt.days
FEAT = ["PM25", "PM10", "NO2", "CO", "O3", "Month"]
m = m[m["gap"] == 1].dropna(subset=FEAT + ["target"])
cut = m["Date"].quantile(0.8)
tr, te = m[m["Date"] <= cut], m[m["Date"] > cut]
rf = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(tr[FEAT], tr["target"])
pred = rf.predict(te[FEAT])
res["model"] = {
    "train_rows": int(len(tr)), "test_rows": int(len(te)), "split_date": str(cut.date()),
    "rf_r2": round(r2_score(te["target"], pred), 3), "rf_mae": round(mean_absolute_error(te["target"], pred), 2),
    "base_r2": round(r2_score(te["target"], te["PM25"]), 3), "base_mae": round(mean_absolute_error(te["target"], te["PM25"]), 2),
    "importance": dict(sorted(zip(FEAT, rf.feature_importances_.round(3).tolist()), key=lambda x: -x[1])),
}

# =====================================================================  4. CHARTS
# ---- Matplotlib
miss = pd.Series(res["raw_missing_pct"]).drop(["City", "Date"], errors="ignore").sort_values()
plt.figure(figsize=(7, 4.5)); plt.barh(miss.index, miss.values, color="#c0562b")
plt.xlabel("Missing values (%)"); plt.title("Missing Data in the Raw Dataset")
fig_done("01_bar_missing", "Bar chart: percentage of missing values per column in the raw data")

top = city.head(10).iloc[::-1]
plt.figure(figsize=(7, 4.5)); plt.barh(top.index, top["mean"], color="#3b6fb6")
plt.xlabel("Mean AQI"); plt.title("Ten Most Polluted Cities by Mean AQI")
fig_done("02_bar_city_aqi", "Bar chart: ten cities with the highest mean AQI")

plt.figure(figsize=(7, 4))
plt.hist(df["PM25"].dropna().clip(upper=df["PM25"].quantile(0.99)), bins=40, color="#4c9f70", edgecolor="white")
plt.axvline(df["PM25"].mean(), color="red", ls="--", label="Mean"); plt.legend()
plt.xlabel("PM2.5 (µg/m³, top 1% clipped)"); plt.ylabel("Days"); plt.title("Distribution of Daily PM2.5")
fig_done("03_hist_pm25", "Histogram: distribution of daily PM2.5 readings")

s = pm_pair.sample(min(5000, len(pm_pair)), random_state=1)
plt.figure(figsize=(6.5, 5)); plt.scatter(s["PM10"], s["PM25"], alpha=0.25, s=10, color="#6f42c1")
k, c0 = np.polyfit(s["PM10"], s["PM25"], 1); xs = np.linspace(0, s["PM10"].max(), 50)
plt.plot(xs, k * xs + c0, color="black", label=f"Trend (slope {k:.2f})"); plt.legend()
plt.xlabel("PM10 (µg/m³)"); plt.ylabel("PM2.5 (µg/m³)"); plt.title("PM2.5 vs PM10")
fig_done("04_scatter_pm", "Scatter plot with trend line: PM2.5 against PM10")

plt.figure(figsize=(5.5, 5.5))
plt.pie(b, labels=b.index, autopct=lambda p: f"{p:.1f}%" if p > 2 else "", startangle=90,
        colors=["#2e7d32", "#9ccc65", "#fdd835", "#fb8c00", "#e53935", "#6a1b9a"])
plt.title("Share of Days by AQI Category")
fig_done("05_pie_aqi_bucket", "Pie chart: share of city-days in each AQI category")

plt.figure(figsize=(7, 4)); plt.plot(monthly.index, monthly.values, marker="o", color="#d9534f")
plt.xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
plt.xlabel("Month"); plt.ylabel("Mean PM2.5 (µg/m³)"); plt.title("Seasonal Cycle of PM2.5")
fig_done("06_line_monthly", "Line chart: mean PM2.5 by calendar month (all cities)")

plt.figure(figsize=(7, 4)); plt.plot(yearly.index, yearly["mean"], marker="o", color="#1f77b4")
plt.xlabel("Year"); plt.ylabel("Mean PM2.5 (µg/m³)"); plt.title("Yearly PM2.5 (average of city means)")
fig_done("07_line_yearly", "Line chart: yearly mean PM2.5 (2020 covers only part of the year)")

# ---- Seaborn
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df.dropna(subset=["PM25"]), x="Season", y="PM25", order=SEASONS, hue="Season", palette="Set2", legend=False, showfliers=False)
plt.ylabel("PM2.5 (µg/m³)"); plt.title("PM2.5 by Season")
fig_done("08_box_season", "Box plot: PM2.5 by season (outliers hidden for readability)")

five = city.head(5).index
plt.figure(figsize=(8, 4.5))
sns.violinplot(data=df[df["City"].isin(five)].dropna(subset=["AQI"]), x="City", y="AQI", order=list(five),
               hue="City", palette="pastel", legend=False, cut=0)
plt.title("AQI Distribution in the Five Most Polluted Cities")
fig_done("09_violin_city", "Violin plot: AQI distribution in the five most polluted cities")

plt.figure(figsize=(9, 7)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, annot_kws={"size": 7})
plt.title("Correlation Between Pollutants and AQI")
fig_done("10_heatmap_corr", "Heatmap: correlation matrix of pollutants and AQI")

plt.figure(figsize=(8, 4.5))
sns.countplot(data=df.dropna(subset=["AQI_Bucket"]), x="Season", hue="AQI_Bucket", order=SEASONS, hue_order=BUCKETS,
              palette=["#2e7d32", "#9ccc65", "#fdd835", "#fb8c00", "#e53935", "#6a1b9a"])
plt.legend(fontsize=7, title="AQI category"); plt.title("AQI Categories by Season")
fig_done("11_count_bucket_season", "Count plot: number of days in each AQI category, by season")

sub = df[["PM25", "PM10", "NO2", "CO"]].dropna().sample(1500, random_state=2)
pp = sns.pairplot(sub, corner=True, plot_kws={"alpha": 0.3, "s": 10}); pp.figure.suptitle("Pair Plot of Key Pollutants", y=1.02)
pp.savefig("figures/12_pairplot.png", dpi=130); plt.close("all")
res["figures"].append({"file": "12_pairplot.png", "caption": "Pair plot: PM2.5, PM10, NO2 and CO"})

top_c = df.dropna(subset=["PM25"]).groupby("City").size().sort_values(ascending=False).head(10).index
hm = df[df["City"].isin(top_c)].pivot_table(index="City", columns="Month", values="PM25", aggfunc="mean")
plt.figure(figsize=(9, 5)); sns.heatmap(hm, cmap="YlOrRd", annot=True, fmt=".0f", annot_kws={"size": 7})
plt.title("Mean PM2.5 by City and Month"); fig_done("13_heatmap_city_month", "Heatmap: mean PM2.5 by city and month (ten cities with most data)")

if "lockdown_change_pct" in res:
    x = np.arange(len(lock)); plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, lock["mean_19"], 0.4, label="2019", color="#8da0cb"); plt.bar(x + 0.2, lock["mean_20"], 0.4, label="2020", color="#fc8d62")
    plt.xticks(x, lock.index, rotation=30, ha="right"); plt.ylabel("Mean PM2.5 (µg/m³)")
    plt.title("PM2.5, 25 Mar - 30 Apr: 2019 vs 2020 Lockdown"); plt.legend()
    fig_done("14_bar_lockdown", "Grouped bar chart: PM2.5 in the lockdown window, 2019 vs 2020")

plt.figure(figsize=(5.5, 5.5)); ss = te.assign(p=pred).sample(min(3000, len(te)), random_state=3)
plt.scatter(ss["target"], ss["p"], alpha=0.3, s=10, color="#2a9d8f")
lim = float(max(ss["target"].max(), ss["p"].max())); plt.plot([0, lim], [0, lim], "k--", label="Perfect prediction")
plt.xlabel("Actual next-day PM2.5"); plt.ylabel("Predicted"); plt.title("Random Forest: Predicted vs Actual"); plt.legend()
fig_done("15_model_pred", "Scatter plot: predicted against actual next-day PM2.5 on the test set")

# =====================================================================  5. FINDINGS
def nm(x):
    return "PM2.5" if x == "PM25" else x


def pfmt(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


mo = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
mdl = res["model"]
F = []
F.append(f"{res['poor_plus_pct']}% of all city-days fall in the Poor, Very Poor or Severe AQI categories, while only {res['bucket_pct']['Good']}% are rated Good.")
F.append(f"{res['worst_city']} has the highest mean AQI ({res['worst_city_aqi']}) among the {res['n_cities_ranked']} cities with at least a year of data, while {res['best_city']} has the lowest ({res['best_city_aqi']}). {res['most_poor_city']} has the largest share of Poor-or-worse days ({res['most_poor_city_pct']}%).")
F.append(f"Pollution varies with the season: mean PM2.5 is highest in {res['worst_season']} and lowest in {res['best_season']}, a {res['season_ratio']}x difference. The worst month on average is {mo[res['worst_month']]} and the best is {mo[res['best_month']]}.")
F.append(f"The seasonal difference is {'statistically significant' if res['kruskal_p'] < 0.05 else 'not statistically significant'} (Kruskal-Wallis H = {res['kruskal_H']}, p {pfmt(res['kruskal_p'])}).")
t = res["top_corr_with_aqi"]
F.append(f"AQI is most strongly correlated with {nm(t[0][0])} (r = {t[0][1]}), followed by {nm(t[1][0])} (r = {t[1][1]}) and {nm(t[2][0])} (r = {t[2][1]}).")
F.append(f"PM2.5 and PM10 move together (Spearman rho = {res['pm25_pm10_spearman']}), and the median PM2.5 reading is about {int(round(res['pm25_pm10_ratio'] * 100))}% of the PM10 reading on the same day.")
yt = res["yearly_table"]
if len(yt) >= 3:
    F.append(f"Average PM2.5 across cities was {yt[0]['pm25']} µg/m³ in {yt[0]['year']} and {yt[-1]['pm25']} µg/m³ in {yt[-1]['year']} (the final year is partial). The number of reporting cities changes over the years, so this trend should be read cautiously.")
if "lockdown_change_pct" in res:
    direction = "fell" if res["lockdown_change_pct"] < 0 else "rose"
    sig = "significant" if res["lockdown_p"] < 0.05 else "not statistically significant"
    F.append(f"In the COVID-19 lockdown window (25 Mar - 30 Apr), mean PM2.5 {direction} by {abs(res['lockdown_change_pct'])}% compared with the same dates in 2019 ({res['lockdown_mean_2019']} to {res['lockdown_mean_2020']} µg/m³; {res['lockdown_cities_improved']} of {res['lockdown_n_cities']} cities improved). A Wilcoxon signed-rank test on city means shows this is {sig} (p {pfmt(res['lockdown_p'])}).")
F.append(f"A random forest predicts next-day PM2.5 with R² = {mdl['rf_r2']} and MAE = {mdl['rf_mae']} µg/m³ on unseen later dates, compared with R² = {mdl['base_r2']} and MAE = {mdl['base_mae']} for the simple 'tomorrow equals today' baseline.")
res["findings"] = F

better = mdl["rf_r2"] > mdl["base_r2"]
tp = res["top_corr_with_aqi"][0][0]
pm_led = tp in ("PM25", "PM10")
res["conclusion"] = [
    f"{res['worst_season']} has the highest PM2.5 levels and {res['best_season']} the lowest, a {res['season_ratio']}x difference (Kruskal-Wallis p {pfmt(res['kruskal_p'])}). {res['poor_plus_pct']}% of recorded city-days were Poor or worse, and {res['worst_city']} was the most polluted city on average (mean AQI {res['worst_city_aqi']}). "
    + ("AQI follows particulate matter most closely." if pm_led else f"AQI was most strongly correlated with {nm(tp)}."),
    "The prediction model " + ("beat" if better else "did not beat") + " the simple 'tomorrow equals today' baseline, which suggests day-to-day PM2.5 is "
    + ("partly predictable from other pollutants and the time of year." if better else "mostly explained by today's PM2.5 alone."),
    "Limitations: the dataset has many missing values (only short gaps were filled), monitoring stations differ between cities and years, and the analysis shows association, not cause. Weather data such as wind and rainfall would improve both the analysis and the model.",
]

with open("results.json", "w") as f:
    json.dump(res, f, indent=2, default=str)

print("Done. Cleaned rows:", res["clean_rows"], "| cities:", res["n_cities"], "|", res["date_min"], "to", res["date_max"])
print("\nFINDINGS")
for i, t in enumerate(F, 1):
    print(f"{i}. {t}")
print("\nNext: python build_report.py")