"""
ماژول آمار توصیفی، جداول فراوانی، شاخص‌های مرکز و پراکندگی و مصورسازی (تسک‌های ۶ تا ۹ فاز اول)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# تنظیم استایل استاندارد علمی برای نمودارها
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def prepare_categorical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    ایجاد برچسب‌های خوانا برای متغیرهای طبقه‌ای و ساخت متغیر ۴ سطحی گروه مصرف دخانیات
    بر روی زیرجمعیت بزرگسالان (۲۰ سال و بالاتر) جهت استفاده در آمار توصیفی تسک‌های ۶ تا ۹.
    """
    df_out = df[df["RIDAGEYR"] >= 20].copy()

    # ۱. برچسب جنسیت
    gender_map = {1.0: "Male", 2.0: "Female"}
    df_out["Gender_Label"] = df_out["RIAGENDR"].map(gender_map)

    # ۲. برچسب سطح تحصیلات بزرگسالان
    educ_map = {
        1.0: "1. Less than 9th grade",
        2.0: "2. 9-11th grade",
        3.0: "3. High school grad/GED",
        4.0: "4. Some college/AA degree",
        5.0: "5. College graduate or above"
    }
    df_out["Education_Label"] = df_out["DMDEDUC2"].map(educ_map)

    # ۳. برچسب نژاد و قومیت
    race_map = {
        1.0: "Mexican American",
        2.0: "Other Hispanic",
        3.0: "Non-Hispanic White",
        4.0: "Non-Hispanic Black",
        6.0: "Non-Hispanic Asian",
        7.0: "Other/Multi-Racial"
    }
    df_out["Race_Label"] = df_out["RIDRETH3"].map(race_map)

    # ۴. اصلاح گمشدگی ساختاری در تعداد افراد سیگاری داخل منزل (SMD470)
    # اگر در خانه کسی سیگار نمی‌کشد، مقدار واقعی صفر نفر است
    df_out["Household_Smokers_Clean"] = df_out["SMD470"].fillna(0.0)
    df_out["Secondhand_Smoke_Binary"] = np.where(
        df_out["Household_Smokers_Clean"] > 0, "Exposed (>=1 smoker)", "Unexposed (0 smokers)"
    )

    # ۵. ساخت متغیر ۴ سطحی گروه مصرف دخانیات (Smoking_Group)
    def assign_group(row):
        # حذف تداخل داروهای ترک نیکوتین، قلیان و تنباکوی بدون دود از گروه‌بندی خالص
        if (row.get("SMQ863") == 1.0) or (row.get("SMQ845") == 1.0) or (row.get("SMQ851") == 1.0):
            return np.nan

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

    df_out["Smoking_Group"] = df_out.apply(assign_group, axis=1)
    return df_out


def generate_categorical_frequency_tables(df_adults: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    تسک ۶ (بخش اول): تولید جداول فراوانی مطلق، درصد معتبر (Valid %) و درصد کل (Total %)
    برای متغیرهای طبقه‌ای اصلی پژوهش.
    """
    cat_vars = {
        "Smoking_Group": "Tobacco Use Group (4 Categories)",
        "Gender_Label": "Gender (RIAGENDR)",
        "Education_Label": "Education Level (DMDEDUC2)",
        "Race_Label": "Race / Hispanic Origin (RIDRETH3)",
        "Secondhand_Smoke_Binary": "Household Secondhand Smoke Exposure (SMD470)",
        "LBDCOTLC": "Serum Cotinine Detection Status (0=Above LOD, 1=Below LOD)",
        "LBDBCDLC": "Blood Cadmium Detection Status (0=Above LOD, 1=Below LOD)"
    }

    tables = {}
    total_rows = len(df_adults)

    for col, title in cat_vars.items():
        if col not in df_adults.columns:
            continue
        counts = df_adults[col].value_counts(dropna=False).sort_index()
        valid_total = df_adults[col].notna().sum()

        rows = []
        cum_valid_pct = 0.0
        for cat, cnt in counts.items():
            is_missing = pd.isna(cat)
            cat_name = "Missing / Unclassified (NaN)" if is_missing else str(cat)
            total_pct = round((cnt / total_rows) * 100, 2)
            valid_pct = np.nan if is_missing else round((cnt / valid_total) * 100, 2)
            if not is_missing:
                cum_valid_pct = round(cum_valid_pct + valid_pct, 2)

            rows.append({
                "Variable": title,
                "Category": cat_name,
                "Frequency (n)": int(cnt),
                "Total_Percent (%)": total_pct,
                "Valid_Percent (%)": valid_pct,
                "Cumulative_Valid_Percent (%)": np.nan if is_missing else cum_valid_pct
            })

        tables[col] = pd.DataFrame(rows)

    return tables


def generate_empirical_quantiles_table(df_adults: pd.DataFrame) -> pd.DataFrame:
    """
    تسک ۶ (بخش دوم): خلاصه توزیع تجربی متغیرهای کمی از طریق صدک‌های تجربی
    (P5, P10, P25, P50, P75, P90, P95).
    """
    quant_vars = [
        ("LBXCOT", "Serum Cotinine (ng/mL)"),
        ("LBXHCOT", "Serum Hydroxycotinine (ng/mL)"),
        ("LBXBCD", "Blood Cadmium (ug/L)"),
        ("LBXBPB", "Blood Lead (ug/dL)"),
        ("LBXBSE", "Blood Selenium (ug/L)"),
        ("LBXHGB", "Hemoglobin (g/dL)"),
        ("LBXWBCSI", "White Blood Cell Count (1000 cells/uL)"),
        ("RIDAGEYR", "Age in Years"),
        ("INDFMPIR", "Family Income-to-Poverty Ratio")
    ]

    records = []
    for col, label in quant_vars:
        s = df_adults[col].dropna()
        q = np.percentile(s, [5, 10, 25, 50, 75, 90, 95])
        records.append({
            "Variable": col,
            "Description": label,
            "Valid_N": len(s),
            "Min": round(float(s.min()), 3),
            "P5": round(float(q[0]), 3),
            "P10": round(float(q[1]), 3),
            "P25 (Q1)": round(float(q[2]), 3),
            "P50 (Median)": round(float(q[3]), 3),
            "P75 (Q3)": round(float(q[4]), 3),
            "P90": round(float(q[5]), 3),
            "P95": round(float(q[6]), 3),
            "Max": round(float(s.max()), 3)
        })

    return pd.DataFrame(records)


def compute_center_and_spread(df_adults: pd.DataFrame) -> pd.DataFrame:
    """
    تسک ۷: محاسبه شاخص‌های مرکز (میانگین، میانه) و پراکندگی (انحراف معیار، چارک‌ها، IQR)
    به همراه ضریب چولگی (Skewness) و تعیین معیار مناسب برای هر متغیر.
    """
    quant_vars = [
        ("LBXCOT", "Serum Cotinine (ng/mL)"),
        ("LBXHCOT", "Serum Hydroxycotinine (ng/mL)"),
        ("LBXBCD", "Blood Cadmium (ug/L)"),
        ("LBXBPB", "Blood Lead (ug/dL)"),
        ("LBXBSE", "Blood Selenium (ug/L)"),
        ("LBXHGB", "Hemoglobin (g/dL)"),
        ("LBXWBCSI", "White Blood Cell Count (1000/uL)"),
        ("RIDAGEYR", "Age (Years)"),
        ("INDFMPIR", "Income-to-Poverty Ratio"),
        ("SMQ720", "Cigarettes/Day in Active Smokers (Last 5d)")
    ]

    rows = []
    for col, desc in quant_vars:
        s = df_adults[col].dropna()
        mean_val = float(s.mean())
        sd_val = float(s.std(ddof=1))
        q1, med, q3 = np.percentile(s, [25, 50, 75])
        iqr_val = float(q3 - q1)
        skew_val = float(s.skew())
        cv_val = (sd_val / mean_val) if mean_val != 0 else np.nan

        if abs(skew_val) > 1.0:
            pref = "Median & IQR (Severe Right Skewness / Outliers)"
        else:
            pref = "Mean & SD (Approximately Symmetric / Bounded)"

        rows.append({
            "Variable": col,
            "Description": desc,
            "N": len(s),
            "Mean": round(mean_val, 3),
            "SD": round(sd_val, 3),
            "Median (Q2)": round(float(med), 3),
            "Q1": round(float(q1), 3),
            "Q3": round(float(q3), 3),
            "IQR": round(iqr_val, 3),
            "Skewness": round(skew_val, 2),
            "CV (SD/Mean)": round(cv_val, 2),
            "Preferred_Metric": pref
        })

    return pd.DataFrame(rows)


def create_phase1_visualizations(df_adults: pd.DataFrame, output_dir: str = "figures") -> List[str]:
    """
    تسک ۸: تولید ۴ نمودار استاندارد، علمی و بدون گمراه‌کنندگی با محورها، واحدها و عناوین دقیق.
    ذخیره فایل‌ها در پوشه figures/.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    df_valid_grp = df_adults[df_adults["Smoking_Group"].notna()].copy()
    group_order = [
        "1. Non-Smoker",
        "2. Exclusive Vape",
        "3. Exclusive Cigarette",
        "4. Dual User"
    ]
    palette_groups = ["#2b5c8f", "#2ca02c", "#d95f02", "#7570b3"]

    # ---------------------------------------------------------
    # نمودار ۱: مقایسه توزیع خام و تبدیل لگاریتمی کوتینین و کادمیوم (Histogram + KDE)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    # الف) کوتینین خام
    s_cot = df_valid_grp["LBXCOT"].dropna()
    sns.histplot(s_cot, bins=50, kde=False, color="#3182bd", ax=axes[0, 0])
    axes[0, 0].axvline(s_cot.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {s_cot.mean():.2f} ng/mL")
    axes[0, 0].axvline(s_cot.median(), color="darkgreen", linestyle="-", linewidth=1.5, label=f"Median: {s_cot.median():.3f} ng/mL")
    axes[0, 0].set_title("A. Raw Serum Cotinine Distribution (Extreme Right Skew)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Serum Cotinine Concentration (ng/mL)", fontsize=10)
    axes[0, 0].set_ylabel("Number of Participants (Count)", fontsize=10)
    axes[0, 0].legend()

    # ب) لگاریتم کوتینین
    s_log_cot = np.log10(s_cot)
    sns.histplot(s_log_cot, bins=40, kde=True, color="#2b5c8f", ax=axes[0, 1])
    axes[0, 1].axvline(np.log10(0.015), color="gray", linestyle=":", linewidth=1.2, label="Laboratory LOD Threshold")
    axes[0, 1].axvline(np.log10(10.0), color="darkred", linestyle="--", linewidth=1.2, label="Active Smoking Cutoff (~10 ng/mL)")
    axes[0, 1].set_title("B. Log10-Transformed Serum Cotinine (Bimodal Regime)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Log10 Serum Cotinine [log10(ng/mL)]", fontsize=10)
    axes[0, 1].set_ylabel("Number of Participants (Count)", fontsize=10)
    axes[0, 1].legend()

    # ج) کادمیوم خام
    s_cd = df_valid_grp["LBXBCD"].dropna()
    sns.histplot(s_cd, bins=50, kde=False, color="#e6550d", ax=axes[1, 0])
    axes[1, 0].axvline(s_cd.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {s_cd.mean():.3f} ug/L")
    axes[1, 0].axvline(s_cd.median(), color="darkgreen", linestyle="-", linewidth=1.5, label=f"Median: {s_cd.median():.3f} ug/L")
    axes[1, 0].set_title("C. Raw Blood Cadmium Distribution (Right Skewed)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Blood Cadmium Concentration (ug/L)", fontsize=10)
    axes[1, 0].set_ylabel("Number of Participants (Count)", fontsize=10)
    axes[1, 0].legend()

    # د) لگاریتم کادمیوم
    s_log_cd = np.log10(s_cd)
    sns.histplot(s_log_cd, bins=35, kde=True, color="#a63603", ax=axes[1, 1])
    axes[1, 1].axvline(s_log_cd.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean Log10: {s_log_cd.mean():.2f}")
    axes[1, 1].set_title("D. Log10-Transformed Blood Cadmium (Near-Normal Shape)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Log10 Blood Cadmium [log10(ug/L)]", fontsize=10)
    axes[1, 1].set_ylabel("Number of Participants (Count)", fontsize=10)
    axes[1, 1].legend()

    fig.suptitle("Figure 1: Empirical Distributions of Primary Biomarkers on Linear vs. Logarithmic Scales (NHANES 2021-2023)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    f1_path = os.path.join(output_dir, "fig1_biomarker_distributions.png")
    plt.savefig(f1_path, bbox_inches="tight")
    plt.show()
    saved_files.append(f1_path)

    # ---------------------------------------------------------
    # نمودار ۲: نمودار جعبه‌ای (Boxplot) کوتینین و کادمیوم به تفکیک ۴ گروه مصرف دخانیات
    # ---------------------------------------------------------
    df_valid_grp["Log10_Cotinine"] = np.log10(df_valid_grp["LBXCOT"])
    df_valid_grp["Log10_Cadmium"] = np.log10(df_valid_grp["LBXBCD"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(
        data=df_valid_grp, x="Smoking_Group", y="Log10_Cotinine",
        order=group_order, palette=palette_groups, width=0.5, ax=axes[0], fliersize=2
    )
    axes[0].set_title("A. Serum Cotinine by Tobacco Use Group (Log10 Scale)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Tobacco Consumption Group", fontsize=10)
    axes[0].set_ylabel("Log10 Serum Cotinine [log10(ng/mL)]", fontsize=10)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    sns.boxplot(
        data=df_valid_grp, x="Smoking_Group", y="Log10_Cadmium",
        order=group_order, palette=palette_groups, width=0.5, ax=axes[1], fliersize=2
    )
    axes[1].set_title("B. Blood Cadmium by Tobacco Use Group (Log10 Scale)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Tobacco Consumption Group", fontsize=10)
    axes[1].set_ylabel("Log10 Blood Cadmium [log10(ug/L)]", fontsize=10)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Figure 2: Comparative Boxplots of Nicotine Exposure and Heavy Metal Burden Across 4 User Groups", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    f2_path = os.path.join(output_dir, "fig2_group_boxplots.png")
    plt.savefig(f2_path, bbox_inches="tight")
    plt.show()
    saved_files.append(f2_path)

    # ---------------------------------------------------------
    # نمودار ۳: توزیع متغیرهای طبقه‌ای (سطح تحصیلات و جنسیت در میان ۴ گروه دخانیات)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # بخش الف: فراوانی مطلق و درصد ۴ گروه دخانیات به تفکیک جنسیت
    ct_gender = pd.crosstab(df_valid_grp["Smoking_Group"], df_valid_grp["Gender_Label"], normalize="index") * 100
    ct_gender = ct_gender.reindex(group_order)
    ct_gender.plot(kind="bar", stacked=False, color=["#e78ac3", "#66c2a5"], edgecolor="black", ax=axes[0])
    axes[0].set_title("A. Gender Composition Within Each Tobacco Group (%)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Tobacco Consumption Group", fontsize=10)
    axes[0].set_ylabel("Percentage within Group (%)", fontsize=10)
    axes[0].set_ylim(0, 75)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(title="Gender")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # بخش ب: ترکیب سطح تحصیلات در میان ۴ گروه دخانیات (نمودار ستونی انباشته ۱۰۰٪)
    ct_educ = pd.crosstab(df_valid_grp["Smoking_Group"], df_valid_grp["Education_Label"], normalize="index") * 100
    ct_educ = ct_educ.reindex(group_order)
    ct_educ.plot(kind="bar", stacked=True, colormap="viridis", edgecolor="black", ax=axes[1])
    axes[1].set_title("B. Educational Attainment Distribution Across Groups (100% Stacked)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Tobacco Consumption Group", fontsize=10)
    axes[1].set_ylabel("Cumulative Percentage (%)", fontsize=10)
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="Education Level", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)

    fig.suptitle("Figure 3: Demographic and Socioeconomic Composition Across Tobacco Consumption Groups", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    f3_path = os.path.join(output_dir, "fig3_demographic_bars.png")
    plt.savefig(f3_path, bbox_inches="tight")
    plt.show()
    saved_files.append(f3_path)

    # ---------------------------------------------------------
    # نمودار ۴: رابطه دوز-پاسخ (تعداد نخ سیگار روزانه با غلظت کادمیوم و کوتینین)
    # ---------------------------------------------------------
    df_smokers = df_valid_grp[
        (df_valid_grp["Smoking_Group"].isin(["3. Exclusive Cigarette", "4. Dual User"])) &
        (df_valid_grp["SMQ720"].notna()) &
        (df_valid_grp["LBXCOT"].notna()) &
        (df_valid_grp["LBXBCD"].notna())
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    sns.regplot(
        data=df_smokers, x="SMQ720", y="LBXCOT",
        scatter_kws={"alpha": 0.35, "color": "#2b5c8f", "s": 25},
        line_kws={"color": "darkred", "linewidth": 2}, ax=axes[0]
    )
    axes[0].set_title("A. Daily Cigarette Dose vs. Serum Cotinine Concentration", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Number of Cigarettes Smoked per Day (Last 5 Days - SMQ720)", fontsize=10)
    axes[0].set_ylabel("Serum Cotinine Concentration (ng/mL)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    sns.regplot(
        data=df_smokers, x="SMQ720", y="LBXBCD",
        scatter_kws={"alpha": 0.35, "color": "#d95f02", "s": 25},
        line_kws={"color": "black", "linewidth": 2}, ax=axes[1]
    )
    axes[1].set_title("B. Daily Cigarette Dose vs. Blood Cadmium Concentration", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Number of Cigarettes Smoked per Day (Last 5 Days - SMQ720)", fontsize=10)
    axes[1].set_ylabel("Blood Cadmium Concentration (ug/L)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Figure 4: Dose-Response Scatter Plots in Active Cigarette Smokers (NHANES 2021-2023)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    f4_path = os.path.join(output_dir, "fig4_dose_response_scatter.png")
    plt.savefig(f4_path, bbox_inches="tight")
    plt.show()
    saved_files.append(f4_path)

    return saved_files


def generate_group_comparison_tables(df_adults: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    تسک ۹: مقایسه توصیفی اولیه متغیرهای کمی (بیومارکرها، سن، درآمد) و متغیرهای طبقه‌ای
    میان ۴ گروه مصرف دخانیات (بدون ادعای استنباطی).
    """
    df_valid = df_adults[df_adults["Smoking_Group"].notna()].copy()
    quant_cols = ["LBXCOT", "LBXHCOT", "LBXBCD", "LBXBPB", "LBXBSE", "LBXHGB", "LBXWBCSI", "RIDAGEYR", "INDFMPIR"]

    comp_rows = []
    for grp, df_g in df_valid.groupby("Smoking_Group"):
        for col in quant_cols:
            s = df_g[col].dropna()
            q1, med, q3 = np.percentile(s, [25, 50, 75]) if len(s) > 0 else (np.nan, np.nan, np.nan)
            comp_rows.append({
                "Smoking_Group": grp,
                "Variable": col,
                "Valid_N": len(s),
                "Mean": round(float(s.mean()), 3) if len(s) > 0 else np.nan,
                "SD": round(float(s.std(ddof=1)), 3) if len(s) > 1 else np.nan,
                "Median": round(float(med), 3),
                "IQR": round(float(q3 - q1), 3)
            })

    df_quant_comp = pd.DataFrame(comp_rows)

    # جدول خلاصه ستونی برای مقایسه سریع چهار گروه در گزارش
    pivot_summary = df_quant_comp.pivot(
        index="Variable", columns="Smoking_Group", values=["Mean", "Median", "IQR"]
    )

    return df_quant_comp, pivot_summary
