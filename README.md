# Air Quality Analysis (Mini Project)

Case study of air pollution across Indian cities (2015-2020) using Pandas, NumPy, Matplotlib, Seaborn, SciPy and scikit-learn.

## Dataset
"Air Quality Data in India" (Rohan Rao, Kaggle), file `city_day.csv`, based on CPCB records.
Download it from Kaggle and place it at `data/city_day.csv`.

## What it does
1. Data cleaning: missing values, duplicates, data types, inconsistent labels, impossible values
2. Analysis: city ranking, seasonal pattern, yearly trend, correlation
3. Statistical tests: seasonal differences, COVID-19 lockdown effect on PM2.5
4. Prediction model: random forest forecasting next-day PM2.5
5. 15 charts and an auto-generated Word report

## How to run
```bash
pip install -r requirements.txt
python analysis.py        # cleans data, makes charts, results.json
python build_report.py    # builds report/Air_Quality_Case_Study.docx
```