"""
ماژول قضیه بیز و الزامات ویژه مسیر الف (NHANES)
شامل: تسک ۹ (به‌روزرسانی بیزی)، الف-۱ (احتمال شرطی در زیرگروه‌ها)،
الف-۲ (مقایسه زیرنمونه‌های هم‌اندازه با مدل دوجمله‌ای)،
الف-۳ (شکل توزیع شاخص سلامت هموگلوبین) و الف-۴ (نقش وزن‌ها در احتمال تجربی)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

GROUP_ORDER = [
    "1. Non-Smoker",
    "2. Exclusive Vape",
    "3. Exclusive Cigarette",
    "4. Dual User"
]


# ==============================================================================
# تسک ۹: قضیه بیز (به‌روزرسانی احتمال پیشین به پسین با فراوانی‌های تجربی)
# ==============================================================================

def compute_bayes_updating(df_p2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    تسک ۹: اجرای به‌روزرسانی بیزی روی افراز چهار گروه مصرف دخانیات (G1 تا G4)
    و رویداد دودویی سیگار احتراقی (A) تحت دو شواهد زیستی:
      سناریو ۱: مشاهده کادمیوم بالای خون (B: LBXBCD >= 0.5 ug/L)
      سناریو ۲: مشاهده نیکوتین فعال همراه با کادمیوم پایین (C ∩ B^c: امضای زیستی ویپ)
    """
    n_total = len(df_p2)

    def _bayes_on_partition(evidence_mask: pd.Series, evidence_name: str) -> pd.DataFrame:
        n_ev_total = int(evidence_mask.sum())
        p_ev_total = n_ev_total / n_total

        rows = []
        for grp in GROUP_ORDER:
            g_mask = df_p2["Smoking_Group"] == grp
            n_g = int(g_mask.sum())
            n_joint = int((g_mask & evidence_mask).sum())

            prior = n_g / n_total
            likelihood = n_joint / n_g if n_g > 0 else 0.0
            joint_prob = prior * likelihood  # معادل n_joint / n_total
            posterior = joint_prob / p_ev_total if p_ev_total > 0 else 0.0
            update_ratio = posterior / prior if prior > 0 else np.nan

            rows.append({
                "Evidence": evidence_name,
                "Group (Hypothesis G_i)": grp,
                "Group_N": n_g,
                "Prior P(G_i)": round(prior, 4),
                "Likelihood P(E | G_i)": round(likelihood, 4),
                "Joint P(G_i n E)": round(joint_prob, 4),
                "Total_Prob P(E)": round(p_ev_total, 4),
                "Posterior P(G_i | E)": round(posterior, 4),
                "Update_Factor (Post/Prior)": round(update_ratio, 2)
            })
        return pd.DataFrame(rows)

    # سناریو ۱: شواهد = کادمیوم بالا (B)
    ev_b = df_p2["Event_B_HighCd"] == 1
    df_bayes_b = _bayes_on_partition(ev_b, "B: High Cadmium (>=0.5)")

    # سناریو ۲: شواهد = نیکوتین فعال (C) و کادمیوم پایین (B^c)
    ev_c_lowcd = (df_p2["Event_C_ActiveCot"] == 1) & (df_p2["Event_B_HighCd"] == 0)
    df_bayes_c_lowcd = _bayes_on_partition(ev_c_lowcd, "C n B^c: Active Nicotine & Low Cd")

    # خلاصه بیزی دودویی برای رویداد A (سیگار احتراقی) به شرط B
    p_a = float(df_p2["Event_A_Combustible"].mean())
    p_not_a = 1.0 - p_a
    p_b_given_a = float(df_p2.loc[df_p2["Event_A_Combustible"] == 1, "Event_B_HighCd"].mean())
    p_b_given_not_a = float(df_p2.loc[df_p2["Event_A_Combustible"] == 0, "Event_B_HighCd"].mean())
    p_b_total = p_a * p_b_given_a + p_not_a * p_b_given_not_a
    p_a_given_b = (p_a * p_b_given_a) / p_b_total

    df_bayes_binary = pd.DataFrame([
        {
            "Hypothesis": "A: Combustible Smoker (G3 U G4)",
            "Prior": round(p_a, 4),
            "Likelihood P(B | H)": round(p_b_given_a, 4),
            "Joint P(H n B)": round(p_a * p_b_given_a, 4),
            "P(B) Law of Total Prob": round(p_b_total, 4),
            "Posterior P(H | B)": round(p_a_given_b, 4),
            "Update_Factor": round(p_a_given_b / p_a, 2)
        },
        {
            "Hypothesis": "A^c: Non-Combustible (G1 U G2)",
            "Prior": round(p_not_a, 4),
            "Likelihood P(B | H)": round(p_b_given_not_a, 4),
            "Joint P(H n B)": round(p_not_a * p_b_given_not_a, 4),
            "P(B) Law of Total Prob": round(p_b_total, 4),
            "Posterior P(H | B)": round(1.0 - p_a_given_b, 4),
            "Update_Factor": round((1.0 - p_a_given_b) / p_not_a, 2)
        }
    ])

    return df_bayes_b, df_bayes_c_lowcd, df_bayes_binary


# ==============================================================================
# الزام الف-۱: احتمال شرطی یک پیامد دودویی در زیرگروه‌های سنی و جنسیتی
# ==============================================================================

def compute_subgroup_conditional_probs(df_p2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    الزام الف-۱: محاسبه توصیفی احتمال شرطی کادمیوم بالا (رویداد B) در زیرگروه‌های
    سنی، جنسیتی و تقاطع سن × جنس (بدون انجام آزمون فرض رسمی).
    """
    df = df_p2.copy()
    if "Gender_Label" not in df.columns:
        df["Gender_Label"] = df["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})

    p_overall = float(df["Event_B_HighCd"].mean())

    def _agg_table(group_cols):
        g = df.groupby(group_cols, observed=False).agg(
            Subgroup_N=("Event_B_HighCd", "size"),
            HighCd_Count=("Event_B_HighCd", "sum"),
            P_B_Given_Subgroup=("Event_B_HighCd", "mean"),
            P_A_Combustible_in_Subgroup=("Event_A_Combustible", "mean")
        ).reset_index()
        g["P(B | Subgroup)"] = g["P_B_Given_Subgroup"].round(4)
        g["P(B | Subgroup) %"] = (g["P_B_Given_Subgroup"] * 100).round(2)
        g["Diff_vs_Overall_P(B)"] = (g["P_B_Given_Subgroup"] - p_overall).round(4)
        g["Smoker_Share_P(A|Sub) %"] = (g["P_A_Combustible_in_Subgroup"] * 100).round(2)
        return g.drop(columns=["P_B_Given_Subgroup", "P_A_Combustible_in_Subgroup"])

    df_by_age = _agg_table(["Age_Group"])
    df_by_gender = _agg_table(["Gender_Label"])
    df_by_age_gender = _agg_table(["Age_Group", "Gender_Label"])

    return df_by_age, df_by_gender, df_by_age_gender


def plot_bayes_and_subgroups(
    df_p2: pd.DataFrame,
    df_bayes_b: pd.DataFrame,
    df_bayes_c_lowcd: pd.DataFrame,
    df_by_age_gender: pd.DataFrame,
    output_dir: str = "figures"
) -> str:
    """
    نمودار دوپنلی برای تسک ۹ (پنل A: به‌روزرسانی بیزی) و الزام الف-۱ (پنل B: زیرگروه‌های سن × جنس).
    """
    os.makedirs(output_dir, exist_ok=True)
    p_b_overall = float(df_p2["Event_B_HighCd"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # پنل A: مقایسه پیشین و دو پسین بیزی
    x = np.arange(len(GROUP_ORDER))
    w = 0.26
    priors = df_bayes_b["Prior P(G_i)"].to_numpy()
    post_b = df_bayes_b["Posterior P(G_i | E)"].to_numpy()
    post_clow = df_bayes_c_lowcd["Posterior P(G_i | E)"].to_numpy()

    b1 = axes[0].bar(x - w, priors, w, label="Prior P(G_i)", color="#7f7f7f", edgecolor="black", linewidth=0.6)
    b2 = axes[0].bar(x, post_b, w, label="Posterior P(G_i | High Cd: B)", color="#d95f02", edgecolor="black", linewidth=0.6)
    b3 = axes[0].bar(x + w, post_clow, w, label="Posterior P(G_i | Active Nic & Low Cd: C n B^c)", color="#2b5c8f", edgecolor="black", linewidth=0.6)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(
                f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8
            )

    axes[0].set_title("A. Bayesian Updating Across 4 Tobacco Groups (Task 9)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["1. Non-Smoker", "2. Excl. Vape", "3. Excl. Cig", "4. Dual User"], fontsize=9.5)
    axes[0].set_ylabel("Probability", fontsize=10)
    axes[0].set_ylim(0, 0.95)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8.2, loc="upper right")

    # پنل B: احتمال شرطی کادمیوم بالا در تقاطع سن و جنسیت
    age_labels = ["1. Young Adult (20-39)", "2. Middle-Aged (40-59)", "3. Older Adult (60+)"]
    short_age = ["Young\n(20-39)", "Middle-Aged\n(40-59)", "Older Adult\n(60+)"]
    female_vals = [
        float(df_by_age_gender.loc[(df_by_age_gender["Age_Group"] == a) & (df_by_age_gender["Gender_Label"] == "Female"), "P(B | Subgroup)"].iloc[0])
        for a in age_labels
    ]
    male_vals = [
        float(df_by_age_gender.loc[(df_by_age_gender["Age_Group"] == a) & (df_by_age_gender["Gender_Label"] == "Male"), "P(B | Subgroup)"].iloc[0])
        for a in age_labels
    ]

    x2 = np.arange(len(age_labels))
    w2 = 0.34
    bf = axes[1].bar(x2 - w2 / 2, female_vals, w2, label="Female", color="#e377c2", edgecolor="black", linewidth=0.6)
    bm = axes[1].bar(x2 + w2 / 2, male_vals, w2, label="Male", color="#1b9e77", edgecolor="black", linewidth=0.6)
    axes[1].axhline(p_b_overall, color="darkred", linestyle="--", linewidth=1.4, label=f"Overall P(B) = {p_b_overall:.3f}")

    for bars in [bf, bm]:
        for bar in bars:
            h = bar.get_height()
            axes[1].annotate(
                f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

    axes[1].set_title("B. Conditional P(High Cadmium) by Age & Gender (Track A-1)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(short_age, fontsize=9.5)
    axes[1].set_ylabel("Empirical Conditional Probability P(LBXBCD >= 0.5)", fontsize=10)
    axes[1].set_ylim(0, max(max(female_vals), max(male_vals)) + 0.08)
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=8.8, loc="upper left")

    fig.suptitle("Bayesian Diagnostic Updating and Subgroup Conditional Probabilities (N = 4,655)", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "p2_fig5_bayes_and_subgroups.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path


# ==============================================================================
# الزام الف-۲: مقایسه شمارش رویداد در زیرنمونه‌های هم‌اندازه با مدل دوجمله‌ای
# ==============================================================================

def compare_subsamples_with_binomial(
    df_p2: pd.DataFrame,
    n_sub: int = 30,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    الزام الف-۲: افراز جمعیت تحلیلی (N=4,655) به زیرنمونه‌های مجزا و هم‌اندازه (M=155 دسته 30 نفری،
    بدون بازنمونه‌گیری/Bootstrap) در دو حالت:
      ۱) افراز تصادفی (بررسی برازش دوجمله‌ای در نبود خوشه‌بندی)
      ۲) افراز پیوسته مرتب‌شده بر اساس خوشه PSU (بررسی اثر همبستگی درون‌خوشه‌ای NHANES)
    و مقایسه توزیع تجربی شمارش‌ها با توزیع نظری Binomial(n=30, p_hat).
    """
    p_hat = float(df_p2["Event_B_HighCd"].mean())
    n_total = len(df_p2)
    m_groups = n_total // n_sub  # 4655 // 30 = 155 زیرنمونه هم‌اندازه مجزا
    used_n = m_groups * n_sub

    # ۱. افراز به ۱۵۵ زیرنمونه مجزای ۳۰ نفری با ترتیب تصادفی ثابت (بدون جایگذاری)
    rng = np.random.default_rng(seed)
    vals_rand = df_p2["Event_B_HighCd"].to_numpy(dtype=int).copy()
    rng.shuffle(vals_rand)
    counts_rand = vals_rand[:used_n].reshape(m_groups, n_sub).sum(axis=1)

    # ۲. افراز به ۱۵۵ زیرنمونه مجزای ۳۰ نفری مرتب‌شده بر اساس طبقه و خوشه PSU
    df_psu_sorted = df_p2.sort_values(["SDMVSTRA", "SDMVPSU", "SEQN"])
    vals_psu = df_psu_sorted["Event_B_HighCd"].to_numpy(dtype=int)
    counts_psu = vals_psu[:used_n].reshape(m_groups, n_sub).sum(axis=1)

    # مشخصات نظری دوجمله‌ای
    theo_mean = n_sub * p_hat
    theo_var = n_sub * p_hat * (1.0 - p_hat)
    theo_sd = np.sqrt(theo_var)

    summary_rows = [
        {
            "Distribution": f"Theoretical Binomial(n={n_sub}, p={p_hat:.4f})",
            "Num_Subsamples (M)": m_groups,
            "Mean": round(theo_mean, 3),
            "Variance": round(theo_var, 3),
            "SD": round(theo_sd, 3),
            "Dispersion_Ratio (Var / Theo_Var)": 1.00,
            "Min": 0,
            "Median": int(stats.binom.ppf(0.50, n_sub, p_hat)),
            "Max": n_sub
        },
        {
            "Distribution": f"Empirical: Random Disjoint Subsamples (n={n_sub}, seed={seed})",
            "Num_Subsamples (M)": m_groups,
            "Mean": round(float(np.mean(counts_rand)), 3),
            "Variance": round(float(np.var(counts_rand, ddof=1)), 3),
            "SD": round(float(np.std(counts_rand, ddof=1)), 3),
            "Dispersion_Ratio (Var / Theo_Var)": round(float(np.var(counts_rand, ddof=1) / theo_var), 2),
            "Min": int(np.min(counts_rand)),
            "Median": round(float(np.median(counts_rand)), 1),
            "Max": int(np.max(counts_rand))
        },
        {
            "Distribution": f"Empirical: PSU-Ordered Disjoint Subsamples (n={n_sub})",
            "Num_Subsamples (M)": m_groups,
            "Mean": round(float(np.mean(counts_psu)), 3),
            "Variance": round(float(np.var(counts_psu, ddof=1)), 3),
            "SD": round(float(np.std(counts_psu, ddof=1)), 3),
            "Dispersion_Ratio (Var / Theo_Var)": round(float(np.var(counts_psu, ddof=1) / theo_var), 2),
            "Min": int(np.min(counts_psu)),
            "Median": round(float(np.median(counts_psu)), 1),
            "Max": int(np.max(counts_psu))
        }
    ]
    df_sub_summary = pd.DataFrame(summary_rows)

    # جدول مقایسه فراوانی تجربی و مورد انتظار دوجمله‌ای در بازه‌های شمارش k
    bins_labels = [
        ("0-3", lambda c: (c >= 0) & (c <= 3), float(stats.binom.cdf(3, n_sub, p_hat))),
        ("4-5", lambda c: (c >= 4) & (c <= 5), float(stats.binom.cdf(5, n_sub, p_hat) - stats.binom.cdf(3, n_sub, p_hat))),
        ("6-7", lambda c: (c >= 6) & (c <= 7), float(stats.binom.cdf(7, n_sub, p_hat) - stats.binom.cdf(5, n_sub, p_hat))),
        ("8-9", lambda c: (c >= 8) & (c <= 9), float(stats.binom.cdf(9, n_sub, p_hat) - stats.binom.cdf(7, n_sub, p_hat))),
        ("10+", lambda c: (c >= 10), float(1.0 - stats.binom.cdf(9, n_sub, p_hat)))
    ]
    pmf_rows = []
    for label, cond_fn, theo_p in bins_labels:
        obs_r = int(np.sum(cond_fn(counts_rand)))
        obs_p = int(np.sum(cond_fn(counts_psu)))
        pmf_rows.append({
            "Count_Bin (k in n=30)": label,
            "Binomial_Theo_Prob": round(theo_p, 4),
            "Expected_Subsamples (of 155)": round(theo_p * m_groups, 1),
            "Obs_Random_Count": obs_r,
            "Obs_Random_Prob": round(obs_r / m_groups, 4),
            "Obs_PSU_Count": obs_p,
            "Obs_PSU_Prob": round(obs_p / m_groups, 4)
        })
    df_pmf_comp = pd.DataFrame(pmf_rows)

    raw_arrays = {"counts_rand": counts_rand, "counts_psu": counts_psu, "p_hat": np.array([p_hat])}
    return df_sub_summary, df_pmf_comp, raw_arrays


def plot_binomial_subsamples(
    raw_arrays: Dict[str, np.ndarray],
    n_sub: int = 30,
    output_dir: str = "figures"
) -> str:
    """
    نمودار الزام الف-۲: مقایسه توزیع تجربی شمارش‌ها در ۱۵۵ زیرنمونه مجزای ۳۰ نفری
    (تصادفی در پنل چپ و مرتب‌شده بر اساس خوشه PSU در پنل راست) با منحنی نظری Binomial(30, p_hat).
    """
    os.makedirs(output_dir, exist_ok=True)
    counts_rand = raw_arrays["counts_rand"]
    counts_psu = raw_arrays["counts_psu"]
    p_hat = float(raw_arrays["p_hat"][0])

    max_k = int(max(counts_rand.max(), counts_psu.max(), 14)) + 1
    k_grid = np.arange(0, max_k + 1)
    theo_pmf = stats.binom.pmf(k_grid, n_sub, p_hat)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)

    # پنل A: افراز تصادفی
    bins = np.arange(-0.5, max_k + 1.5, 1.0)
    axes[0].hist(
        counts_rand, bins=bins, density=True, color="#3182bd", alpha=0.65,
        edgecolor="black", linewidth=0.6, label=f"Empirical Random Subsamples (Var={np.var(counts_rand, ddof=1):.2f})"
    )
    axes[0].plot(
        k_grid, theo_pmf, color="darkred", marker="o", markersize=5, linewidth=2.0,
        label=f"Theoretical Binomial(30, {p_hat:.3f}) (Var={n_sub*p_hat*(1-p_hat):.2f})"
    )
    axes[0].set_title("A. Random Disjoint Subsamples (M=155, n=30)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("High-Cadmium Count per Subsample (k)", fontsize=10)
    axes[0].set_ylabel("Relative Frequency / Probability Mass", fontsize=10)
    axes[0].set_xlim(-0.5, max_k + 0.5)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8.5, loc="upper right")

    # پنل B: افراز مرتب‌شده بر اساس PSU
    axes[1].hist(
        counts_psu, bins=bins, density=True, color="#e6550d", alpha=0.65,
        edgecolor="black", linewidth=0.6, label=f"Empirical PSU-Ordered Subsamples (Var={np.var(counts_psu, ddof=1):.2f})"
    )
    axes[1].plot(
        k_grid, theo_pmf, color="black", marker="o", markersize=5, linewidth=2.0,
        label=f"Theoretical Binomial(30, {p_hat:.3f}) (Var={n_sub*p_hat*(1-p_hat):.2f})"
    )
    axes[1].set_title("B. PSU-Clustered Disjoint Subsamples (M=155, n=30)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("High-Cadmium Count per Subsample (k)", fontsize=10)
    axes[1].set_xlim(-0.5, max_k + 0.5)
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Empirical Subsample Counts vs. Binomial Model: Random vs. Clustered Partitioning (Track A-2)", fontsize=12.2, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "p2_fig6_binomial_subsamples.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path


# ==============================================================================
# الزام الف-۳: شکل توزیع یک اندازه‌گیری سلامت (هموگلوبین خون LBXHGB)
# ==============================================================================

def analyze_health_biomarker_shape(df_p2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    الزام الف-۳: تحلیل کامل چولگی، کشیدگی، نقاط دورافتاده (قاعده 1.5*IQR توکی) و تقریب نرمال
    برای شاخص بالینی هموگلوبین خون (LBXHGB) در کل جمعیت تحلیلی و به تفکیک جنسیت.
    """
    df = df_p2.copy()
    if "Gender_Label" not in df.columns:
        df["Gender_Label"] = df["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})

    def _profile(series: pd.Series, label: str) -> Dict[str, object]:
        s = series.dropna().to_numpy(dtype=float)
        mu = float(np.mean(s))
        sd = float(np.std(s, ddof=1))
        q1, med, q3 = np.percentile(s, [25, 50, 75])
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        ext_low_fence = q1 - 3.0 * iqr

        n_low_out = int(np.sum(s < low_fence))
        n_high_out = int(np.sum(s > high_fence))
        n_ext_low = int(np.sum(s < ext_low_fence))

        return {
            "Group": label,
            "Valid_N": len(s),
            "Mean": round(mu, 3),
            "Median": round(float(med), 3),
            "SD": round(sd, 3),
            "Q1": round(float(q1), 3),
            "Q3": round(float(q3), 3),
            "IQR": round(float(iqr), 3),
            "Skewness": round(float(stats.skew(s, bias=False)), 3),
            "Excess_Kurtosis": round(float(stats.kurtosis(s, fisher=True, bias=False)), 3),
            "Within_1SD (%)": round(float(np.mean(np.abs(s - mu) <= sd) * 100), 2),
            "Within_2SD (%)": round(float(np.mean(np.abs(s - mu) <= 2 * sd) * 100), 2),
            "Within_3SD (%)": round(float(np.mean(np.abs(s - mu) <= 3 * sd) * 100), 2),
            "Lower_Fence": round(float(low_fence), 2),
            "Upper_Fence": round(float(high_fence), 2),
            "Low_Outliers (n)": n_low_out,
            "High_Outliers (n)": n_high_out,
            "Total_Outliers (%)": round(((n_low_out + n_high_out) / len(s)) * 100, 2),
            "Extreme_Low_3IQR (n)": n_ext_low
        }

    overall_row = _profile(df["LBXHGB"], "Overall Analytic Cohort")
    female_row = _profile(df.loc[df["Gender_Label"] == "Female", "LBXHGB"], "Female Only")
    male_row = _profile(df.loc[df["Gender_Label"] == "Male", "LBXHGB"], "Male Only")

    df_hgb_profile = pd.DataFrame([overall_row, female_row, male_row])

    # مقایسه صدک‌های تجربی با صدک‌های نظری نرمال برای کل جمعیت تحلیلی
    s_all = df["LBXHGB"].dropna().to_numpy(dtype=float)
    mu_a, sd_a = np.mean(s_all), np.std(s_all, ddof=1)
    probs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    q_rows = []
    for p in probs:
        eq = float(np.percentile(s_all, p * 100))
        tq = float(stats.norm.ppf(p, loc=mu_a, scale=sd_a))
        q_rows.append({
            "Percentile": f"P{int(p*100)}",
            "Empirical_LBXHGB (g/dL)": round(eq, 3),
            "Normal_Theoretical (g/dL)": round(tq, 3),
            "Diff (Emp - Theo)": round(eq - tq, 3)
        })
    df_hgb_quantiles = pd.DataFrame(q_rows)

    return df_hgb_profile, df_hgb_quantiles


def plot_hemoglobin_shape(df_p2: pd.DataFrame, output_dir: str = "figures") -> str:
    """
    نمودار چهارپنلی الزام الف-۳ برای هموگلوبین خون (LBXHGB):
      A: هیستوگرام + منحنی نرمال + مرزهای نقاط پرت توکی
      B: نمودار Q-Q نرمال
      C: نمودار جعبه‌ای برای نمایش صریح نقاط دورافتاده دنباله چپ و راست
      D: تفکیک توزیع بر حسب جنسیت (تبیین فیزیولوژیک چولگی خفیف منفی در کل نمونه)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = df_p2.copy()
    if "Gender_Label" not in df.columns:
        df["Gender_Label"] = df["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})

    s = df["LBXHGB"].dropna().to_numpy(dtype=float)
    mu, sd = np.mean(s), np.std(s, ddof=1)
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    low_f, high_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))

    # پنل A: هیستوگرام و منحنی نرمال
    x_grid = np.linspace(s.min() - 0.5, s.max() + 0.5, 400)
    axes[0, 0].hist(s, bins=38, density=True, color="#2b5c8f", alpha=0.65, edgecolor="black", linewidth=0.4, label="Empirical Density")
    axes[0, 0].plot(x_grid, stats.norm.pdf(x_grid, mu, sd), color="darkred", linewidth=2.0, label=f"Normal PDF N({mu:.2f}, {sd:.2f}^2)")
    axes[0, 0].axvline(low_f, color="#d95f02", linestyle="--", linewidth=1.3, label=f"Tukey Fences [{low_f:.2f}, {high_f:.2f}]")
    axes[0, 0].axvline(high_f, color="#d95f02", linestyle="--", linewidth=1.3)
    axes[0, 0].set_title("A. Hemoglobin Distribution vs. Normal Curve", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Hemoglobin Concentration (g/dL)", fontsize=10)
    axes[0, 0].set_ylabel("Probability Density", fontsize=10)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0, 0].legend(fontsize=8.3, loc="upper left")

    # پنل B: نمودار Q-Q
    (osm, osr), (slope, intercept, _) = stats.probplot(s, dist="norm")
    axes[0, 1].scatter(osm, osr, s=11, alpha=0.45, color="#2b5c8f", label="Sample Quantiles")
    axes[0, 1].plot(osm, slope * osm + intercept, color="darkred", linestyle="--", linewidth=1.8, label="Normal Reference Line")
    axes[0, 1].set_title("B. Normal Q-Q Plot (Left Tail Deviation / Anemia Cases)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Theoretical Standard Normal Quantiles (z)", fontsize=10)
    axes[0, 1].set_ylabel("Observed Hemoglobin (g/dL)", fontsize=10)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend(fontsize=8.5, loc="upper left")

    # پنل C: نمودار جعبه‌ای افقی نقاط پرت
    axes[1, 0].boxplot(
        s, vert=False, widths=0.45, patch_artist=True,
        boxprops=dict(facecolor="#6baed6", edgecolor="black"),
        medianprops=dict(color="darkred", linewidth=2.0),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="#d95f02", alpha=0.65, markeredgecolor="none")
    )
    axes[1, 0].axvline(low_f, color="#d95f02", linestyle=":", linewidth=1.2)
    axes[1, 0].axvline(high_f, color="#d95f02", linestyle=":", linewidth=1.2)
    axes[1, 0].set_title("C. Boxplot & Tukey 1.5*IQR Outliers (Concentrated in Left Tail)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Hemoglobin Concentration (g/dL)", fontsize=10)
    axes[1, 0].set_yticks([1])
    axes[1, 0].set_yticklabels(["LBXHGB\n(N=4,638)"])
    axes[1, 0].grid(axis="x", linestyle="--", alpha=0.4)

    # پنل D: تفکیک بر حسب جنسیت (توضیح زیستی شکل توزیع)
    s_f = df.loc[df["Gender_Label"] == "Female", "LBXHGB"].dropna().to_numpy(dtype=float)
    s_m = df.loc[df["Gender_Label"] == "Male", "LBXHGB"].dropna().to_numpy(dtype=float)
    kde_f = stats.gaussian_kde(s_f)
    kde_m = stats.gaussian_kde(s_m)
    axes[1, 1].hist(s_f, bins=32, density=True, alpha=0.38, color="#e377c2", label=f"Female (Mean={np.mean(s_f):.2f}, SD={np.std(s_f, ddof=1):.2f})")
    axes[1, 1].hist(s_m, bins=32, density=True, alpha=0.38, color="#1b9e77", label=f"Male (Mean={np.mean(s_m):.2f}, SD={np.std(s_m, ddof=1):.2f})")
    axes[1, 1].plot(x_grid, kde_f(x_grid), color="#c51b8a", linewidth=2.0)
    axes[1, 1].plot(x_grid, kde_m(x_grid), color="#006837", linewidth=2.0)
    axes[1, 1].set_title("D. Sex-Stratified Mixture Structure of Hemoglobin", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Hemoglobin Concentration (g/dL)", fontsize=10)
    axes[1, 1].set_ylabel("Probability Density", fontsize=10)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1, 1].legend(fontsize=8.5, loc="upper left")

    fig.suptitle("Distributional Shape, Outliers, and Normal Approximation of Hemoglobin (Track A-3)", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "p2_fig7_hemoglobin_shape.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path


# ==============================================================================
# الزام الف-۴: نقش وزن‌های نمونه‌گیری در احتمال تجربی (خام در برابر وزن‌دار)
# ==============================================================================

def compare_unweighted_vs_weighted_probabilities(df_p2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    الزام الف-۴: مقایسه فراوانی نسبی خام در نمونه (Unweighted Sample Frequency)
    با برآورد احتمال در جمعیت (Weighted Population Probability با WTPH2YR)
    برای رویدادهای اصلی فاز ۲ و احتمال شرطی کادمیوم بالا در چهار گروه.
    """
    w = df_p2["WTPH2YR"].to_numpy(dtype=float)
    w_sum = float(np.sum(w))
    n_total = len(df_p2)

    # بخش ۱: رویدادهای حاشیه‌ای کلیدی
    event_defs = [
        ("B: High Blood Cadmium (LBXBCD >= 0.5 ug/L)", df_p2["Event_B_HighCd"].to_numpy(dtype=float)),
        ("A: Combustible Smoker (G3 U G4)", df_p2["Event_A_Combustible"].to_numpy(dtype=float)),
        ("C: Active Nicotine (LBXCOT >= 10.0 ng/mL)", df_p2["Event_C_ActiveCot"].to_numpy(dtype=float)),
        ("G2: Exclusive Vape User", (df_p2["Smoking_Group"] == "2. Exclusive Vape").to_numpy(dtype=float)),
        ("Older Adult (Age >= 60 Years - Design Driver)", (df_p2["RIDAGEYR"] >= 60).to_numpy(dtype=float))
    ]

    marg_rows = []
    for name, ind in event_defs:
        p_unw = float(np.mean(ind))
        p_w = float(np.sum(w * ind) / w_sum)
        diff = p_w - p_unw
        rel_pct = (diff / p_unw) * 100 if p_unw > 0 else np.nan
        marg_rows.append({
            "Event": name,
            "Sample_Count (n)": int(np.sum(ind)),
            "Unweighted_Prob (Sample Freq)": round(p_unw, 4),
            "Weighted_Prob (Pop Estimate)": round(p_w, 4),
            "Diff (W - Unw)": round(diff, 4),
            "Relative_Change (%)": round(rel_pct, 2)
        })
    df_weighted_marginal = pd.DataFrame(marg_rows)

    # بخش ۲: مقایسه احتمال شرطی P(B | G_i) در حالت خام و وزن‌دار
    cond_rows = []
    for grp in GROUP_ORDER:
        sub = df_p2[df_p2["Smoking_Group"] == grp]
        ind_b = sub["Event_B_HighCd"].to_numpy(dtype=float)
        w_sub = sub["WTPH2YR"].to_numpy(dtype=float)
        p_unw = float(np.mean(ind_b))
        p_w = float(np.sum(w_sub * ind_b) / np.sum(w_sub))
        cond_rows.append({
            "Smoking_Group": grp,
            "Group_N": len(sub),
            "Unweighted P(B | G_i)": round(p_unw, 4),
            "Weighted P_w(B | G_i)": round(p_w, 4),
            "Diff (W - Unw)": round(p_w - p_unw, 4),
            "Relative_Change (%)": round(((p_w - p_unw) / p_unw) * 100, 2)
        })
    df_weighted_cond = pd.DataFrame(cond_rows)

    return df_weighted_marginal, df_weighted_cond
