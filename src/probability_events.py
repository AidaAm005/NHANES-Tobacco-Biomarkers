"""
ماژول رویدادهای احتمالی، احتمال‌های حاشیه‌ای، مشترک، شرطی و بررسی استقلال
(فاز دوم: تسک‌های ۱ تا ۴)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def add_phase2_events_and_rvs(df_analytic: pd.DataFrame) -> pd.DataFrame:
    """
    تسک ۱: افزودن ستون‌های رویدادهای دودویی (A, B, C)، زیرگروه‌های جمعیتی
    و متغیرهای تصادفی پیوسته تبدیل‌یافته روی جمعیت تحلیلی نهایی (df_analytic).
    """
    df = df_analytic.copy()

    # رویداد A: مصرف فعال سیگار احتراقی (G3 U G4)
    df["Event_A_Combustible"] = df["Smoking_Group"].isin(
        ["3. Exclusive Cigarette", "4. Dual User"]
    ).astype(int)
    df["Event_A_Label"] = np.where(
        df["Event_A_Combustible"] == 1, "A: Combustible Smoker", "A^c: Non-Combustible"
    )

    # رویداد B: غلظت بالای کادمیوم خون (LBXBCD >= 0.5 ug/L)
    df["Event_B_HighCd"] = (df["LBXBCD"] >= 0.5).astype(int)
    df["Event_B_Label"] = np.where(
        df["Event_B_HighCd"] == 1, "B: High Cadmium (>=0.5)", "B^c: Low Cadmium (<0.5)"
    )

    # رویداد C: مواجهه فعال با نیکوتین (LBXCOT >= 10.0 ng/mL)
    df["Event_C_ActiveCot"] = (df["LBXCOT"] >= 10.0).astype(int)
    df["Event_C_Label"] = np.where(
        df["Event_C_ActiveCot"] == 1, "C: Active Nicotine (>=10)", "C^c: No Active Nicotine (<10)"
    )

    # رویداد جنسیت (مرد = 1، زن = 0) برای بررسی استقلال در تسک ۴ و زیرگروه در الف-۱
    df["Event_Male"] = (df["RIAGENDR"] == 1.0).astype(int)

    # افراز سنی برای زیرگروه‌ها
    df["Age_Group"] = pd.cut(
        df["RIDAGEYR"],
        bins=[19, 39, 59, 120],
        labels=["1. Young Adult (20-39)", "2. Middle-Aged (40-59)", "3. Older Adult (60+)"]
    )

    # متغیرهای تصادفی پیوسته لگاریتمی (Y1 و Y2)
    df["Log10_LBXBCD"] = np.log10(df["LBXBCD"])
    df["Log10_LBXCOT"] = np.log10(df["LBXCOT"])

    return df


def verify_task1_analytic_summary(df_p2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    راستی‌آزمایی دقیق اعداد و بازه‌های متغیرهای تصادفی و رویدادها مستقیماً روی
    جمعیت تحلیلی نهایی (N = 4,655) برای اطمینان از تطابق ۱۰۰ درصدی تسک ۱.
    """
    # الف) خلاصه فراوانی رویدادهای دودویی
    event_specs = [
        ("Event_A_Combustible", "A: Combustible Smoker (G3 U G4)"),
        ("Event_B_HighCd", "B: High Blood Cadmium (LBXBCD >= 0.5 ug/L)"),
        ("Event_C_ActiveCot", "C: Active Nicotine (LBXCOT >= 10.0 ng/mL)"),
        ("Event_Male", "M: Male Participants (RIAGENDR = 1)")
    ]
    n_total = len(df_p2)
    ev_rows = []
    for col, desc in event_specs:
        cnt = int(df_p2[col].sum())
        ev_rows.append({
            "Event": desc,
            "Count (n)": cnt,
            "Complement_Count": n_total - cnt,
            "Total_N": n_total,
            "Empirical_Prob": round(cnt / n_total, 4),
            "Percent (%)": round((cnt / n_total) * 100, 2)
        })
    df_events_summary = pd.DataFrame(ev_rows)

    # ب) خلاصه دامنه دقیق متغیرهای پیوسته در جمعیت تحلیلی نهایی
    rv_cols = [
        ("LBXBCD", "W: Blood Cadmium (ug/L)"),
        ("Log10_LBXBCD", "Y1: Log10 Blood Cadmium"),
        ("LBXCOT", "Raw Serum Cotinine (ng/mL)"),
        ("Log10_LBXCOT", "Y2: Log10 Serum Cotinine"),
        ("LBXHGB", "Y3: Hemoglobin (g/dL)"),
        ("RIDAGEYR", "Age (Years)")
    ]
    rv_rows = []
    for col, desc in rv_cols:
        s = df_p2[col].dropna()
        q1, med, q3 = np.percentile(s, [25, 50, 75])
        rv_rows.append({
            "Variable": col,
            "Description": desc,
            "Valid_N": len(s),
            "Min": round(float(s.min()), 3),
            "Q1": round(float(q1), 3),
            "Median": round(float(med), 3),
            "Q3": round(float(q3), 3),
            "Max": round(float(s.max()), 3)
        })
    df_rvs_summary = pd.DataFrame(rv_rows)

    return df_events_summary, df_rvs_summary


def compute_marginal_and_joint_tables(
    df_p2: pd.DataFrame,
    row_label_col: str,
    col_label_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    تسک ۲: ساخت جدول فراوانی دوطرفه و جدول احتمال‌های مشترک و حاشیه‌ای
    برای دو رویداد طبقه‌ای (مطابق فصل ۳ ووناکات).
    """
    ct_counts = pd.crosstab(
        df_p2[row_label_col],
        df_p2[col_label_col],
        margins=True,
        margins_name="Total (Marginal)"
    )

    ct_probs = pd.crosstab(
        df_p2[row_label_col],
        df_p2[col_label_col],
        normalize="all",
        margins=True,
        margins_name="Total (Marginal)"
    ).round(4)

    return ct_counts, ct_probs


def compute_conditional_probabilities(df_p2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    تسک ۳: محاسبه احتمال‌های شرطی رویدادهای کادمیوم بالا (B) و نیکوتین فعال (C)
    به شرط مصرف سیگار احتراقی (A در برابر A^c) و به تفکیک هر چهار گروه افراز (G1 تا G4).
    """
    p_b_marginal = float(df_p2["Event_B_HighCd"].mean())
    p_c_marginal = float(df_p2["Event_C_ActiveCot"].mean())

    # بخش ۱: شرط کردن روی رویداد دودویی A (سیگار احتراقی در برابر غیر احتراقی)
    binary_rows = []
    for label, sub in df_p2.groupby("Event_A_Label"):
        n_sub = len(sub)
        n_b = int(sub["Event_B_HighCd"].sum())
        n_c = int(sub["Event_C_ActiveCot"].sum())
        p_b_cond = n_b / n_sub
        p_c_cond = n_c / n_sub
        binary_rows.append({
            "Conditioning_Event": label,
            "Group_N": n_sub,
            "Count_B (High Cd)": n_b,
            "P(B | Condition)": round(p_b_cond, 4),
            "Shift_vs_P(B)": round(p_b_cond - p_b_marginal, 4),
            "Ratio_to_P(B)": round(p_b_cond / p_b_marginal, 2),
            "Count_C (Active Cot)": n_c,
            "P(C | Condition)": round(p_c_cond, 4),
            "Shift_vs_P(C)": round(p_c_cond - p_c_marginal, 4)
        })
    df_cond_binary = pd.DataFrame(binary_rows)

    # بخش ۲: شرط کردن روی افراز چهارگانه گروه مصرف دخانیات (G1 تا G4)
    group_order = [
        "1. Non-Smoker",
        "2. Exclusive Vape",
        "3. Exclusive Cigarette",
        "4. Dual User"
    ]
    group_rows = []
    for grp in group_order:
        sub = df_p2[df_p2["Smoking_Group"] == grp]
        n_sub = len(sub)
        n_b = int(sub["Event_B_HighCd"].sum())
        n_c = int(sub["Event_C_ActiveCot"].sum())
        p_b_cond = n_b / n_sub
        p_c_cond = n_c / n_sub
        group_rows.append({
            "Smoking_Group": grp,
            "Group_N": n_sub,
            "Count_B (High Cd)": n_b,
            "P(B | Group)": round(p_b_cond, 4),
            "P(B | Group) %": round(p_b_cond * 100, 2),
            "Count_C (Active Cot)": n_c,
            "P(C | Group)": round(p_c_cond, 4),
            "P(C | Group) %": round(p_c_cond * 100, 2)
        })
    df_cond_4groups = pd.DataFrame(group_rows)

    return df_cond_binary, df_cond_4groups


def evaluate_independence(
    df_p2: pd.DataFrame,
    event1_col: str,
    event2_col: str,
    pair_name: str
) -> Dict[str, object]:
    """
    تسک ۴: ارزیابی استدلالی استقلال دو رویداد از طریق مقایسه:
      1) P(E1 n E2) در برابر P(E1) * P(E2)
      2) P(E2 | E1) در برابر P(E2)
    (مطابق فصل ۳ ووناکات و دستورالعمل فاز ۲، بدون انجام آزمون فرض رسمی).
    """
    n_total = len(df_p2)
    e1 = df_p2[event1_col] == 1
    e2 = df_p2[event2_col] == 1

    p_e1 = float(e1.mean())
    p_e2 = float(e2.mean())
    p_joint_obs = float((e1 & e2).mean())
    p_joint_indep = p_e1 * p_e2

    p_e2_given_e1 = float(df_p2.loc[e1, event2_col].mean())
    p_e2_given_not_e1 = float(df_p2.loc[~e1, event2_col].mean())

    diff_joint = p_joint_obs - p_joint_indep
    ratio_joint = p_joint_obs / p_joint_indep if p_joint_indep > 0 else np.nan
    diff_cond = p_e2_given_e1 - p_e2

    return {
        "Event_Pair": pair_name,
        "N_Total": n_total,
        "P(E1)": round(p_e1, 4),
        "P(E2)": round(p_e2, 4),
        "Observed_P(E1 n E2)": round(p_joint_obs, 4),
        "Expected_Under_Indep_P(E1)*P(E2)": round(p_joint_indep, 4),
        "Joint_Diff (Obs - Exp)": round(diff_joint, 4),
        "Joint_Ratio (Obs / Exp)": round(ratio_joint, 2),
        "P(E2 | E1)": round(p_e2_given_e1, 4),
        "P(E2 | E1^c)": round(p_e2_given_not_e1, 4),
        "Cond_Shift [P(E2|E1) - P(E2)]": round(diff_cond, 4)
    }


def plot_conditional_probabilities(
    df_p2: pd.DataFrame,
    df_cond_4groups: pd.DataFrame,
    output_dir: str = "figures"
) -> str:
    """
    رسم نمودار مقایسه‌ای احتمال شرطی مواجهه فعال با نیکوتین P(C | G_i)
    در برابر احتمال شرطی کادمیوم بالای خون P(B | G_i) در ۴ گروه مصرف دخانیات.
    """
    os.makedirs(output_dir, exist_ok=True)

    groups = df_cond_4groups["Smoking_Group"].tolist()
    p_c_vals = df_cond_4groups["P(C | Group)"].tolist()
    p_b_vals = df_cond_4groups["P(B | Group)"].tolist()

    p_c_marg = float(df_p2["Event_C_ActiveCot"].mean())
    p_b_marg = float(df_p2["Event_B_HighCd"].mean())

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars1 = ax.bar(
        x - width / 2, p_c_vals, width,
        label="P(Active Nicotine: LBXCOT >= 10 ng/mL | Group)",
        color="#2b5c8f", edgecolor="black", linewidth=0.7
    )
    bars2 = ax.bar(
        x + width / 2, p_b_vals, width,
        label="P(High Cadmium: LBXBCD >= 0.5 ug/L | Group)",
        color="#d95f02", edgecolor="black", linewidth=0.7
    )

    # خطوط افقی احتمال حاشیه‌ای کل جمعیت تحلیلی
    ax.axhline(
        p_c_marg, color="#2b5c8f", linestyle="--", linewidth=1.3, alpha=0.85,
        label=f"Overall Marginal P(C) = {p_c_marg:.3f}"
    )
    ax.axhline(
        p_b_marg, color="#d95f02", linestyle=":", linewidth=1.6, alpha=0.9,
        label=f"Overall Marginal P(B) = {p_b_marg:.3f}"
    )

    # درج مقادیر دقیق بالای میله‌ها
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#1b3b5c"
        )
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#9c4200"
        )

    ax.set_title(
        "Conditional Probabilities of Active Nicotine vs. High Blood Cadmium Across 4 Tobacco Groups",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Tobacco Consumption Group (Partition G1 - G4)", fontsize=10.5)
    ax.set_ylabel("Empirical Conditional Probability", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8.8, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "p2_fig1_conditional_probs.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path
