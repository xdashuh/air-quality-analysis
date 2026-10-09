"""
Builds report/Air_Quality_Case_Study.docx from results.json and figures/.
Run AFTER analysis.py:   python build_report.py
Needs: pip install python-docx
"""
import os
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("report", exist_ok=True)

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

R = json.load(open("results.json"))
doc = Document()

# ---------- page + base style
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)  # A4
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Inches(0.9))
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)
for name, size in (("Heading 1", 16), ("Heading 2", 13)):
    st = doc.styles[name]
    st.font.name, st.font.size, st.font.bold = "Calibri", Pt(size), True
    st.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def para(text="", bold=False, italic=False, align=None, size=None, after=6):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    return p


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ""
        run = c.paragraphs[0].add_run(str(h)); run.bold = True; run.font.size = Pt(10)
        shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(10)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(file, caption, width=5.6):
    doc.add_picture(f"figures/{file}", width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    n = [f["file"] for f in R["figures"]].index(file) + 1
    para(f"Figure {n}: {caption}", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=9, after=10)


def code(lines):
    for l in lines:
        p = doc.add_paragraph(); r = p.add_run(l); r.font.name = "Consolas"; r.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(0)
    doc.add_paragraph()


def pretty(x):
    return "PM2.5" if x == "PM25" else x


# ---------- title page
for _ in range(6):
    doc.add_paragraph()
para("Mini Project", align=WD_ALIGN_PARAGRAPH.CENTER, size=18)
para("Case Study: Air Quality Analysis", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=26)
para("Air pollution across Indian cities, 2015 to 2020", align=WD_ALIGN_PARAGRAPH.CENTER, size=14)
para("Python: Pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, after=40)
for line in ("Name: ____________________________", "Roll No.: ________________________",
             "Course / Section: _________________", "Submission date: ____________________"):
    para(line, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ---------- 1
doc.add_heading("1. Problem Statement", 1)
para("Air pollution is a serious public-health problem in India. To reduce it, we first need to understand where and when it is worst, "
     "which pollutants drive it, and whether it is changing over time.")
para("This case study investigates: How do air pollutants vary over time and across cities? The analysis covers AQI trends, "
     "pollutant correlation and seasonal analysis. Three extra questions are also examined: did the 2020 COVID-19 lockdown "
     "change PM2.5 levels, are seasonal differences statistically significant, and can tomorrow's PM2.5 be predicted?")

# ---------- 2
doc.add_heading("2. Dataset Description", 1)
para(f"Dataset: \"Air Quality Data in India (2015 - 2020)\" (file city_day.csv), published on Kaggle by Rohan Rao and compiled from "
     f"Central Pollution Control Board (CPCB) monitoring records. It contains one row per city per day: {R['raw_rows']:,} records and "
     f"{R['raw_cols']} columns, covering {R['n_cities']} cities from {R['date_min']} to {R['date_max']}.")
table(["Column", "Type", "Description"], [
    ["City, Date", "Text, Date", "Monitoring city and day"],
    ["PM2.5, PM10", "Numeric", "Fine and coarse particulate matter (µg/m³)"],
    ["NO, NO2, NOx, NH3", "Numeric", "Nitrogen compounds and ammonia (µg/m³)"],
    ["CO", "Numeric", "Carbon monoxide (mg/m³)"],
    ["SO2, O3", "Numeric", "Sulphur dioxide and ozone (µg/m³)"],
    ["Benzene, Toluene, Xylene", "Numeric", "Volatile organic compounds (µg/m³)"],
    ["AQI", "Numeric", "Air Quality Index for the day"],
    ["AQI_Bucket", "Categorical", "Good, Satisfactory, Moderate, Poor, Very Poor, Severe"],
], [1.9, 1.2, 3.5])
para("Columns added during analysis: Year, Month, Season (Winter Dec-Feb, Summer Mar-May, Monsoon Jun-Sep, Post-monsoon Oct-Nov).")

# ---------- 3
doc.add_heading("3. Data Cleaning Using Pandas", 1)
miss = sorted(R["raw_missing_pct"].items(), key=lambda kv: -kv[1])[:5]
fills = R["filled_by_interpolation"]
table(["Issue", "Found in raw data", "Fix applied"], [
    ["Missing values", "Highest: " + ", ".join(f"{pretty(k)} {v}%" for k, v in miss),
     f"Short gaps (up to 7 days) filled by interpolation within each city (PM2.5: {fills.get('PM25', 0):,} values filled). Longer gaps left empty and ignored by each calculation."],
    ["Duplicate records", f"{R['raw_duplicates']} repeated city-date rows", "drop_duplicates() on City and Date"],
    ["Incorrect data types", f"Date stored as text ({R['bad_dates']} unreadable dates)", "pd.to_datetime() and pd.to_numeric()"],
    ["Inconsistent values", "AQI_Bucket text with stray spaces or different capitalisation; missing buckets",
     f"strip() and title-case; {R['bucket_filled']:,} missing buckets rebuilt from the AQI value using official bands"],
    ["Impossible values", ("Negative concentrations: " + ", ".join(f"{pretty(k)} ({v})" for k, v in R["negative_values"].items())) if R["negative_values"] else "None found",
     "Negative readings set to missing. High spikes were kept because they are real pollution events."],
], [1.4, 2.4, 2.8])
para(f"After cleaning: {R['clean_rows']:,} rows. The cleaned file is saved as data/city_day_clean.csv.")
para("Code excerpt:", bold=True)
code(['df["Date"] = pd.to_datetime(df["Date"], errors="coerce")',
      'df = df.drop_duplicates(subset=["City", "Date"])',
      'df.loc[df[c] < 0, c] = np.nan',
      'df[c] = df.groupby("City")[c].transform(lambda s: s.interpolate(limit=7, limit_area="inside"))'])

# ---------- 4
doc.add_heading("4. Data Analysis Using NumPy & Pandas", 1)
doc.add_heading("4.1 Mean, median and standard deviation", 2)
d = R["describe"]
table(["Variable", "Mean", "Median", "Std. dev."], [[pretty(k), d[k]["mean"], d[k]["median"], d[k]["std"]] for k in d], [2.2, 1.4, 1.4, 1.6])
para(f"Using NumPy directly, mean PM2.5 is {R['np_pm25_mean']} µg/m³ with a standard deviation of {R['np_pm25_std']}. "
     "The standard deviation is large compared with the mean, which shows that pollution levels swing widely from day to day.")

doc.add_heading("4.2 Grouping and aggregation", 2)
para(f"Cities with at least a year of AQI records were ranked by mean AQI ({R['n_cities_ranked']} cities qualify). The ten highest:")
table(["City", "Mean AQI", "Days of data"], [[c["city"], c["aqi"], f"{c['days']:,}"] for c in R["city_table"]], [2.5, 1.6, 1.8])
para("Mean PM2.5 by season:")
table(["Season", "Mean PM2.5", "Median PM2.5"], [[s["season"], s["mean"], s["median"]] for s in R["season_table"]], [2.5, 1.6, 1.8])
para("Yearly PM2.5 (average of city means; each city-year needs at least 100 days of data):")
table(["Year", "Mean PM2.5", "Cities included"], [[y["year"], y["pm25"], y["cities"]] for y in R["yearly_table"]], [2.0, 1.8, 2.0])

doc.add_heading("4.3 Sorting and filtering", 2)
para(f"Sorting by mean AQI gives {R['worst_city']} ({R['worst_city_aqi']}) as the most polluted city and {R['best_city']} ({R['best_city_aqi']}) as the cleanest. "
     f"Filtering for days rated Poor, Very Poor or Severe shows they make up {R['poor_plus_pct']}% of all records; the city with the largest share is "
     f"{R['most_poor_city']} ({R['most_poor_city_pct']}% of its days).")
table(["AQI category", "City-days", "Share (%)"], [[k, f"{v:,}", R["bucket_pct"][k]] for k, v in R["bucket_counts"].items()], [2.5, 1.8, 1.6])

doc.add_heading("4.4 Correlation", 2)
para("Correlation of each pollutant with AQI (Pearson, using available data):")
table(["Pollutant", "Correlation with AQI"], [[pretty(k), v] for k, v in R["corr_aqi"].items()], [2.5, 2.5])
a, b, v = R["strongest_pair"]
para(f"The strongest pair of pollutants is {pretty(a)} and {pretty(b)} (r = {v}). PM2.5 and PM10 have a Spearman correlation of {R['pm25_pm10_spearman']}.")

doc.add_heading("4.5 Statistical tests", 2)
rows = [["Kruskal-Wallis: PM2.5 across four seasons", f"H = {R['kruskal_H']}", "< 0.001" if R["kruskal_p"] < 0.001 else f"{R['kruskal_p']:.3f}"],
        ["Spearman: PM2.5 vs PM10", f"rho = {R['pm25_pm10_spearman']}", "< 0.001" if R["pm25_pm10_p"] < 0.001 else f"{R['pm25_pm10_p']:.3f}"]]
if "lockdown_p" in R:
    rows.append([f"Wilcoxon signed-rank: city PM2.5, lockdown window 2019 vs 2020 ({R['lockdown_n_cities']} cities)",
                 f"{R['lockdown_mean_2019']} to {R['lockdown_mean_2020']} ({R['lockdown_change_pct']}%)",
                 "< 0.001" if R["lockdown_p"] < 0.001 else f"{R['lockdown_p']:.3f}"])
table(["Test", "Result", "p-value"], rows, [3.4, 1.9, 1.2])
para("A p-value below 0.05 is conventionally treated as statistically significant. Non-parametric tests were used because pollution data is heavily skewed, not normally distributed.")

doc.add_heading("4.6 Prediction model", 2)
m = R["model"]
para(f"A random forest regression model predicts the next day's PM2.5 using today's PM2.5, PM10, NO2, CO, O3 and the month. "
     f"Rows up to {m['split_date']} were used for training ({m['train_rows']:,} rows) and later rows for testing ({m['test_rows']:,} rows), so the model is judged on dates it has never seen.")
table(["Model", "R²", "MAE (µg/m³)"], [["Baseline: tomorrow = today", m["base_r2"], m["base_mae"]], ["Random forest", m["rf_r2"], m["rf_mae"]]], [3.0, 1.5, 1.8])
para("Feature importance: " + ", ".join(f"{pretty(k)} {v}" for k, v in m["importance"].items()) + ".")

# ---------- 5
doc.add_heading("5. Data Visualization", 1)
mpl = [f for f in R["figures"] if int(f["file"][:2]) in (1, 2, 3, 4, 5, 6, 7)]
snb = [f for f in R["figures"] if int(f["file"][:2]) >= 8 and int(f["file"][:2]) <= 13]
oth = [f for f in R["figures"] if int(f["file"][:2]) >= 14]
doc.add_heading("5.1 Matplotlib", 2)
for f in mpl:
    figure(f["file"], f["caption"], 5.0 if f["file"].startswith("05") else 5.6)
doc.add_heading("5.2 Seaborn", 2)
for f in snb:
    figure(f["file"], f["caption"], 5.8 if f["file"].startswith(("10", "13")) else 5.4)
doc.add_heading("5.3 Lockdown and model charts", 2)
for f in oth:
    figure(f["file"], f["caption"], 5.0)

# ---------- 6
doc.add_heading("6. Findings", 1)
for t in R["findings"]:
    doc.add_paragraph(t, style="List Number")

# ---------- 7
doc.add_heading("7. Conclusion", 1)
for t in R["conclusion"]:
    para(t)
para("Source code, data and charts are available in the accompanying GitHub repository.", italic=True)

out = "report/Air_Quality_Case_Study.docx"
doc.save(out)
print("Saved", out)