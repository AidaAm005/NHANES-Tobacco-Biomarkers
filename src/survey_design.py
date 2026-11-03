"""
ماژول تحلیل طرح نمونه‌گیری، مقایسه آماره‌های خام و وزن‌دار و تعریف جمعیت تحلیلی
(الزامات ویژه مسیر الف: تسک‌های الف-۲، الف-۳ و الف-۴)
"""
import numpy as np
import pandas as pd
from typing import Tuple


def summarize_sampling_weights(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    محاسبه شاخص‌های توصیفی وزن‌های نمونه‌گیری (WTINT2YR, WTMEC2YR, WTPH2YR)
    در میان بزرگسالان جهت نشان دادن اینکه هر فرد نماینده چند نفر در جامعه آمریکا است (مکمل تسک الف-۲).
    """
    df_adults = df_cleaned[df_cleaned["RIDAGEYR"] >= 20].copy()
    weight_cols = [
        ("WTINT2YR", "Interview 2-Year Weight (All Interviewed Adults)"),
        ("WTMEC2YR", "MEC Exam 2-Year Weight (Examined Adults)"),
        ("WTPH2YR", "Phlebotomy 2-Year Weight (Blood Draw Subsample)")
    ]

    rows = []
    for col, desc in weight_cols:
        s = df_adults.loc[df_adults[col] > 0, col].dropna()
        rows.append({
            "Weight_Variable": col,
            "Description": desc,
            "Positive_Weights_N": len(s),
            "Sum_of_Weights (Estimated US Adult Pop)": int(round(s.sum())),
            "Min_Weight (Min Persons Represented)": round(float(s.min()), 1),
            "Median_Weight": round(float(s.median()), 1),
            "Mean_Weight": round(float(s.mean()), 1),
            "Max_Weight (Max Persons Represented)": round(float(s.max()), 1),
            "Max_to_Min_Ratio": round(float(s.max() / s.min()), 1)
        })

    return pd.DataFrame(rows)


def define_analytic_population(df_cleaned: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    تسک الف-۴: اعمال گام‌به‌گام معیارهای ورود و خروج (Inclusion/Exclusion Criteria)،
    ساخت جمعیت تحلیلی نهایی و گزارش تعداد دقیق افراد حذف‌شده در هر مرحله.
    """
    steps_log = []

    # مرحله ۰: کل افراد موجود در فایل ادغام‌شده اولیه
    n0 = len(df_cleaned)
    steps_log.append({
        "Step_Number": 0,
        "Criterion_Description": "Total participants in merged NHANES 2021-2023 dataset (DEMO_L base)",
        "Remaining_N": n0,
        "Excluded_in_Step": 0,
        "Cumulative_Excluded": 0,
        "Retention_Rate (%)": 100.00
    })

    # مرحله ۱: محدود کردن به بزرگسالان ۲۰ سال و بالاتر
    df_s1 = df_cleaned[df_cleaned["RIDAGEYR"] >= 20].copy()
    n1 = len(df_s1)
    steps_log.append({
        "Step_Number": 1,
        "Criterion_Description": "Target age restriction: Adults aged >= 20 years (exclude < 20 years)",
        "Remaining_N": n1,
        "Excluded_in_Step": n0 - n1,
        "Cumulative_Excluded": n0 - n1,
        "Retention_Rate (%)": round((n1 / n0) * 100, 2)
    })

    # مرحله ۲: داشتن وزن خون‌گیری مثبت و معتبر (WTPH2YR > 0)
    df_s2 = df_s1[(df_s1["WTPH2YR"].notna()) & (df_s1["WTPH2YR"] > 0)].copy()
    n2 = len(df_s2)
    steps_log.append({
        "Step_Number": 2,
        "Criterion_Description": "Valid Phlebotomy sampling weight (WTPH2YR > 0; attended MEC blood draw)",
        "Remaining_N": n2,
        "Excluded_in_Step": n1 - n2,
        "Cumulative_Excluded": n0 - n2,
        "Retention_Rate (%)": round((n2 / n0) * 100, 2)
    })

    # مرحله ۳: موجود بودن نتیجه آزمایشگاهی برای هر دو بیومارکر پاسخ اصلی (کوتینین و کادمیوم)
    df_s3 = df_s2[(df_s2["LBXCOT"].notna()) & (df_s2["LBXBCD"].notna())].copy()
    n3 = len(df_s3)
    steps_log.append({
        "Step_Number": 3,
        "Criterion_Description": "Non-missing primary outcomes: Serum Cotinine (LBXCOT) and Blood Cadmium (LBXBCD)",
        "Remaining_N": n3,
        "Excluded_in_Step": n2 - n3,
        "Cumulative_Excluded": n0 - n3,
        "Retention_Rate (%)": round((n3 / n0) * 100, 2)
    })

    # مرحله ۴: حذف مصرف‌کنندگان محصولات تداخلی در ۵ روز اخیر (داروهای ترک نیکوتین NRT، قلیان، تنباکوی بدون دود)
    cond_confound = (df_s3["SMQ863"] == 1.0) | (df_s3["SMQ845"] == 1.0) | (df_s3["SMQ851"] == 1.0)
    df_s4 = df_s3[~cond_confound].copy()
    n4 = len(df_s4)
    steps_log.append({
        "Step_Number": 4,
        "Criterion_Description": "Exclude recent NRT (SMQ863=1), Hookah (SMQ845=1), or Smokeless tobacco (SMQ851=1) users",
        "Remaining_N": n4,
        "Excluded_in_Step": n3 - n4,
        "Cumulative_Excluded": n0 - n4,
        "Retention_Rate (%)": round((n4 / n0) * 100, 2)
    })

    # مرحله ۵: طبقه‌بندی در یکی از ۴ گروه مصرف دخانیات و حذف رکوردهای دارای نقص در پرسش‌نامه دخانیات
    def classify_group(row):
        curr_cig = (row.get("SMQ040") in [1.0, 2.0]) or (row.get("SMQ681") == 1.0)
        no_cig = ((row.get("SMQ020") == 2.0) or (row.get("SMQ040") == 3.0)) and (row.get("SMQ681") == 2.0)
        vape_yes = (row.get("SMQ846") == 1.0)
        vape_no = (row.get("SMQ846") == 2.0)

        if curr_cig and vape_yes:
            return "4. Dual User"
        elif curr_cig and vape_no:
            return "3. Exclusive Cigarette"
        elif no_cig and vape_yes:
            return "2. Exclusive Vape"
        elif no_cig and vape_no:
            return "1. Non-Smoker"
        return np.nan

    df_s4["Smoking_Group"] = df_s4.apply(classify_group, axis=1)
    df_analytic = df_s4[df_s4["Smoking_Group"].notna()].copy()
    n5 = len(df_analytic)

    steps_log.append({
        "Step_Number": 5,
        "Criterion_Description": "Complete smoking/vaping questionnaire data to assign 1 of 4 study groups (Final Analytic Sample)",
        "Remaining_N": n5,
        "Excluded_in_Step": n4 - n5,
        "Cumulative_Excluded": n0 - n5,
        "Retention_Rate (%)": round((n5 / n0) * 100, 2)
    })

    # افزودن برچسب‌های کمکی به دیتافریم تحلیلی نهایی
    df_analytic["Gender_Label"] = df_analytic["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})
    df_analytic["Household_Smokers_Clean"] = df_analytic["SMD470"].fillna(0.0)

    df_flow_report = pd.DataFrame(steps_log)
    return df_analytic, df_flow_report


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """محاسبه صدک وزن‌دار (مثلاً میانه وزن‌دار با q=0.50)."""
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    cum_w = np.cumsum(w_sorted)
    cutoff = q * np.sum(w_sorted)
    return float(v_sorted[np.searchsorted(cum_w, cutoff)])


def compare_unweighted_vs_weighted(df_analytic: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    تسک الف-۳: مقایسه خلاصه‌های آماری خام (Unweighted) و وزن‌دار (Weighted با WTPH2YR)
    برای متغیرهای کمی (میانگین و میانه) و متغیرهای طبقه‌ای (نسبت‌ها/درصدها).
    """
    # الف) مقایسه میانگین و میانه برای متغیرهای کمی
    quant_vars = [
        ("RIDAGEYR", "Age (Years)"),
        ("INDFMPIR", "Income-to-Poverty Ratio"),
        ("LBXCOT", "Serum Cotinine (ng/mL) - All"),
        ("LBXBCD", "Blood Cadmium (ug/L) - All"),
        ("LBXBPB", "Blood Lead (ug/dL) - All"),
        ("LBXHGB", "Hemoglobin (g/dL)"),
        ("LBXWBCSI", "White Blood Cell Count (1000/uL)")
    ]

    quant_rows = []
    for col, desc in quant_vars:
        sub = df_analytic[[col, "WTPH2YR"]].dropna()
        x = sub[col].to_numpy(dtype=float)
        w = sub["WTPH2YR"].to_numpy(dtype=float)

        unw_mean = float(np.mean(x))
        w_mean = float(np.average(x, weights=w))
        diff_mean = w_mean - unw_mean
        rel_diff_pct = (diff_mean / unw_mean) * 100 if unw_mean != 0 else np.nan

        unw_med = float(np.median(x))
        w_med = _weighted_quantile(x, w, 0.50)

        quant_rows.append({
            "Variable": col,
            "Description": desc,
            "Valid_N": len(sub),
            "Unweighted_Mean": round(unw_mean, 3),
            "Weighted_Mean": round(w_mean, 3),
            "Mean_Diff (W - Unw)": round(diff_mean, 3),
            "Relative_Change (%)": round(rel_diff_pct, 2),
            "Unweighted_Median": round(unw_med, 3),
            "Weighted_Median": round(w_med, 3)
        })

    # افزودن میانگین کوتینین و کادمیوم به تفکیک گروه‌های کلیدی (سیگاری و ویپر)
    for grp_name in ["2. Exclusive Vape", "3. Exclusive Cigarette", "4. Dual User"]:
        sub_g = df_analytic[df_analytic["Smoking_Group"] == grp_name]
        for col, short_name in [("LBXCOT", "Cotinine"), ("LBXBCD", "Cadmium")]:
            x = sub_g[col].dropna().to_numpy(dtype=float)
            w = sub_g.loc[sub_g[col].notna(), "WTPH2YR"].to_numpy(dtype=float)
            unw_m = float(np.mean(x))
            w_m = float(np.average(x, weights=w))
            quant_rows.append({
                "Variable": f"{col} ({grp_name})",
                "Description": f"{short_name} in {grp_name}",
                "Valid_N": len(x),
                "Unweighted_Mean": round(unw_m, 3),
                "Weighted_Mean": round(w_m, 3),
                "Mean_Diff (W - Unw)": round(w_m - unw_m, 3),
                "Relative_Change (%)": round(((w_m - unw_m) / unw_m) * 100, 2),
                "Unweighted_Median": round(float(np.median(x)), 3),
                "Weighted_Median": round(_weighted_quantile(x, w, 0.50), 3)
            })

    df_quant_comp = pd.DataFrame(quant_rows)

    # ب) مقایسه نسبت‌ها (درصدها) برای متغیرهای طبقه‌ای
    cat_rows = []
    total_n = len(df_analytic)
    total_w = df_analytic["WTPH2YR"].sum()

    # ۱. چهار گروه مصرف دخانیات
    for grp, df_g in df_analytic.groupby("Smoking_Group"):
        unw_pct = (len(df_g) / total_n) * 100
        w_pct = (df_g["WTPH2YR"].sum() / total_w) * 100
        cat_rows.append({
            "Category_Group": "Tobacco Use Group",
            "Category_Level": grp,
            "Sample_Count (n)": len(df_g),
            "Unweighted_Percent (%)": round(unw_pct, 2),
            "Weighted_Percent (%)": round(w_pct, 2),
            "Difference (W - Unw %)": round(w_pct - unw_pct, 2)
        })

    # ۲. گروه سنی سالمندان (۶۰ سال و بالاتر) برای نشان دادن اصلاح بیش‌نمونه‌گیری سالمندان
    n_seniors = int((df_analytic["RIDAGEYR"] >= 60).sum())
    unw_sen_pct = (n_seniors / total_n) * 100
    w_sen_pct = (df_analytic.loc[df_analytic["RIDAGEYR"] >= 60, "WTPH2YR"].sum() / total_w) * 100
    cat_rows.append({
        "Category_Group": "Age Bracket",
        "Category_Level": "Older Adults (Aged >= 60 Years)",
        "Sample_Count (n)": n_seniors,
        "Unweighted_Percent (%)": round(unw_sen_pct, 2),
        "Weighted_Percent (%)": round(w_sen_pct, 2),
        "Difference (W - Unw %)": round(w_sen_pct - unw_sen_pct, 2)
    })

    # ۳. نژاد سفیدپوست غیرهیسپانیک و سیاه‌پوست غیرهیسپانیک
    for code, r_label in [(3.0, "Non-Hispanic White"), (4.0, "Non-Hispanic Black")]:
        sub_r = df_analytic[df_analytic["RIDRETH3"] == code]
        unw_r = (len(sub_r) / total_n) * 100
        w_r = (sub_r["WTPH2YR"].sum() / total_w) * 100
        cat_rows.append({
            "Category_Group": "Race / Ethnicity",
            "Category_Level": r_label,
            "Sample_Count (n)": len(sub_r),
            "Unweighted_Percent (%)": round(unw_r, 2),
            "Weighted_Percent (%)": round(w_r, 2),
            "Difference (W - Unw %)": round(w_r - unw_r, 2)
        })

    df_cat_comp = pd.DataFrame(cat_rows)
    return df_quant_comp, df_cat_comp
