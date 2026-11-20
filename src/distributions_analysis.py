"""
ماژول تحلیل توزیع‌های گسسته (دوجمله‌ای)، توزیع پیوسته و تقریب نرمال،
امید ریاضی و واریانس، و کوواریانس دو متغیره (فاز دوم: تسک‌های ۵ تا ۸)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, List

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


# ==============================================================================
# تسک ۵: توزیع گسسته و مدل دوجمله‌ای (بررسی فرض‌ها و تقریب)
# ==============================================================================

def analyze_binomial_model_and_assumptions(
    df_p2: pd.DataFrame,
    n_trials: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    تسک ۵: محاسبه مشخصات مدل دوجمله‌ای برای شمارش رویداد B (کادمیوم >= 0.5)
    در n آزمایش (n=30) و ممیزی کمی چهار فرض مدل دوجمله‌ای در NHANES.
    """
    n_total = len(df_p2)
    p_overall = float(df_p2["Event_B_HighCd"].mean())

    # بخش ۱: مقایسه مشخصات مدل Binomial(n=30, p) در کل جمعیت تحلیلی و گروه‌های همگن
    regimes = [
        ("Overall Analytic Population (Omega)", df_p2),
        ("1. Non-Smoker (G1 - Homogeneous)", df_p2[df_p2["Smoking_Group"] == "1. Non-Smoker"]),
        ("2. Exclusive Vape (G2 - Small Subgroup)", df_p2[df_p2["Smoking_Group"] == "2. Exclusive Vape"]),
        ("3. Exclusive Cigarette (G3 - Homogeneous)", df_p2[df_p2["Smoking_Group"] == "3. Exclusive Cigarette"]),
        ("4. Dual User (G4 - Small Subgroup)", df_p2[df_p2["Smoking_Group"] == "4. Dual User"])
    ]

    model_rows = []
    for name, sub in regimes:
        N_sub = len(sub)
        p_hat = float(sub["Event_B_HighCd"].mean())
        sampling_frac = n_trials / N_sub
        fpc = (N_sub - n_trials) / (N_sub - 1)  # ضریب تصحیح جامعه متناهی (فوق‌هندسی در برابر دوجمله‌ای)

        mean_bin = n_trials * p_hat
        var_bin = n_trials * p_hat * (1.0 - p_hat)
        sd_bin = np.sqrt(var_bin)
        var_hyper = var_bin * fpc

        p_zero = float(stats.binom.pmf(0, n_trials, p_hat))
        p_le_5 = float(stats.binom.cdf(5, n_trials, p_hat))
        p_ge_10 = float(1.0 - stats.binom.cdf(9, n_trials, p_hat))

        model_rows.append({
            "Population_Regime": name,
            "Subpop_Size (N)": N_sub,
            "Sampling_Frac (n/N %)": round(sampling_frac * 100, 2),
            "p_hat = P(B)": round(p_hat, 4),
            "E[K_30] = np": round(mean_bin, 2),
            "Var_Binom": round(var_bin, 2),
            "Var_Hypergeom": round(var_hyper, 2),
            "SD_Binom": round(sd_bin, 2),
            "P(K=0)": round(p_zero, 4),
            "P(K<=5)": round(p_le_5, 4),
            "P(K>=10)": round(p_ge_10, 4)
        })
    df_binom_summary = pd.DataFrame(model_rows)

    # بخش ۲: بررسی ناهمگنی احتمال موفقیت P(B) در میان ۳۰ خوشه جغرافیایی (PSU)
    df_psu = df_p2.dropna(subset=["SDMVSTRA", "SDMVPSU"]).copy()
    df_psu["PSU_ID"] = (
        df_psu["SDMVSTRA"].astype(int).astype(str) + "_PSU" + df_psu["SDMVPSU"].astype(int).astype(str)
    )
    psu_stats = df_psu.groupby("PSU_ID").agg(
        PSU_N=("Event_B_HighCd", "size"),
        HighCd_Count=("Event_B_HighCd", "sum"),
        P_B_in_PSU=("Event_B_HighCd", "mean")
    ).reset_index()

    psu_summary = pd.DataFrame([{
        "Num_PSUs": len(psu_stats),
        "Overall_P(B)": round(p_overall, 4),
        "Min_P(B)_Across_PSUs": round(float(psu_stats["P_B_in_PSU"].min()), 4),
        "Max_P(B)_Across_PSUs": round(float(psu_stats["P_B_in_PSU"].max()), 4),
        "Range_Across_PSUs": round(float(psu_stats["P_B_in_PSU"].max() - psu_stats["P_B_in_PSU"].min()), 4),
        "SD_of_P(B)_Across_PSUs": round(float(psu_stats["P_B_in_PSU"].std(ddof=1)), 4)
    }])

    return df_binom_summary, psu_summary, psu_stats


def plot_binomial_model_and_clusters(
    df_p2: pd.DataFrame,
    psu_stats: pd.DataFrame,
    n_trials: int = 30,
    output_dir: str = "figures"
) -> str:
    """
    نمودار تسک ۵:
      پنل A: تابع جرم احتمال نظری Binomial(30, p) در کل جمعیت در برابر ویپرها و سیگاری‌ها
      پنل B: تغییرات احتمال تجربی P(B) در میان ۳۰ خوشه جغرافیایی (PSU) برای نمایش اثر خوشه‌بندی
    """
    os.makedirs(output_dir, exist_ok=True)

    p_all = float(df_p2["Event_B_HighCd"].mean())
    p_vape = float(df_p2.loc[df_p2["Smoking_Group"] == "2. Exclusive Vape", "Event_B_HighCd"].mean())
    p_cig = float(df_p2.loc[df_p2["Smoking_Group"] == "3. Exclusive Cigarette", "Event_B_HighCd"].mean())

    k_vals = np.arange(0, n_trials + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # پنل A: مقایسه PMF دوجمله‌ای
    axes[0].bar(
        k_vals, stats.binom.pmf(k_vals, n_trials, p_all),
        alpha=0.65, color="#3182bd", edgecolor="black", linewidth=0.5,
        label=f"Overall Sample (p = {p_all:.3f}, E[K] = {n_trials*p_all:.1f})"
    )
    axes[0].plot(
        k_vals, stats.binom.pmf(k_vals, n_trials, p_vape),
        marker="o", markersize=4.5, linewidth=1.8, color="#2ca02c",
        label=f"Exclusive Vape (p = {p_vape:.3f}, E[K] = {n_trials*p_vape:.1f})"
    )
    axes[0].plot(
        k_vals, stats.binom.pmf(k_vals, n_trials, p_cig),
        marker="s", markersize=4.5, linewidth=1.8, color="#d95f02",
        label=f"Exclusive Cigarette (p = {p_cig:.3f}, E[K] = {n_trials*p_cig:.1f})"
    )
    axes[0].set_title(f"A. Theoretical Binomial(n={n_trials}, p) PMF Across Regimes", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(f"Count of High-Cadmium Adults in Sample of n={n_trials} (k)", fontsize=10)
    axes[0].set_ylabel("Probability Mass P(K = k)", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8.5, loc="upper center")

    # پنل B: پراکندگی P(B) در میان ۳۰ خوشه PSU
    psu_sorted = psu_stats.sort_values("P_B_in_PSU").reset_index(drop=True)
    x_idx = np.arange(1, len(psu_sorted) + 1)
    axes[1].scatter(x_idx, psu_sorted["P_B_in_PSU"], color="#7570b3", s=42, zorder=3, label="Observed P(B) in Each PSU")
    axes[1].vlines(x_idx, p_all, psu_sorted["P_B_in_PSU"], color="#7570b3", alpha=0.4, linewidth=1.2)
    axes[1].axhline(p_all, color="darkred", linestyle="--", linewidth=1.6, label=f"Overall Marginal P(B) = {p_all:.3f}")
    axes[1].set_title("B. Cluster-Level Variation of P(B) Across 30 NHANES PSUs", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Primary Sampling Units (Sorted 1 to 30)", fontsize=10)
    axes[1].set_ylabel("Empirical Proportion P(LBXBCD >= 0.5) per PSU", fontsize=10)
    axes[1].set_ylim(0, max(0.40, float(psu_sorted["P_B_in_PSU"].max()) + 0.05))
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=9, loc="upper left")

    fig.suptitle("Binomial Model Regimes and Cluster-Level Assumption Audit (NHANES 2021-2023)", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "p2_fig2_binomial_model.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path


# ==============================================================================
# تسک ۶: متغیر پیوسته و بررسی تقریب توزیع نرمال
# ==============================================================================

def evaluate_normal_approximation(df_p2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    تسک ۶: ارزیابی جامع تقریب نرمال برای کادمیوم خام (LBXBCD) و کادمیوم لگاریتمی (Log10_LBXBCD)
    از طریق چولگی، کشیدگی مازاد، قانون تجربی 68-95-99.7%، احتمال منفی ناممکن، و مقایسه صدک‌ها.
    """
    target_vars = [
        ("LBXBCD", "W: Raw Blood Cadmium (ug/L)"),
        ("Log10_LBXBCD", "Y1: Log10 Blood Cadmium [log10(ug/L)]")
    ]

    shape_rows = []
    quantile_rows = []

    for col, desc in target_vars:
        s = df_p2[col].dropna().to_numpy(dtype=float)
        n = len(s)
        mu = float(np.mean(s))
        sd = float(np.std(s, ddof=1))
        med = float(np.median(s))
        skew_val = float(stats.skew(s, bias=False))
        kurt_val = float(stats.kurtosis(s, fisher=True, bias=False))  # کشیدگی مازاد (نرمال = 0)

        # انطباق با قانون تجربی 68.27% - 95.45% - 99.73%
        within_1sd = float(np.mean(np.abs(s - mu) <= 1.0 * sd) * 100)
        within_2sd = float(np.mean(np.abs(s - mu) <= 2.0 * sd) * 100)
        within_3sd = float(np.mean(np.abs(s - mu) <= 3.0 * sd) * 100)

        # بررسی احتمال منفی ناممکن برای مقیاس خام (تنها برای W معنادار است)
        prob_neg_normal = float(stats.norm.cdf(0.0, loc=mu, scale=sd) * 100) if col == "LBXBCD" else np.nan

        shape_rows.append({
            "Variable": desc,
            "N": n,
            "Mean": round(mu, 4),
            "Median": round(med, 4),
            "SD": round(sd, 4),
            "Skewness": round(skew_val, 3),
            "Excess_Kurtosis": round(kurt_val, 3),
            "Within_1SD (%) [Ref: 68.27]": round(within_1sd, 2),
            "Within_2SD (%) [Ref: 95.45]": round(within_2sd, 2),
            "Within_3SD (%) [Ref: 99.73]": round(within_3sd, 2),
            "Normal_P(X<0) (%)": round(prob_neg_normal, 2) if not np.isnan(prob_neg_normal) else np.nan
        })

        # مقایسه ۵ صدک کلیدی تجربی با صدک نظری توزیع نرمال N(mu, sd^2)
        probs = [0.05, 0.25, 0.50, 0.75, 0.95]
        emp_q = np.percentile(s, [p * 100 for p in probs])
        theo_q = stats.norm.ppf(probs, loc=mu, scale=sd)
        for p, eq, tq in zip(probs, emp_q, theo_q):
            quantile_rows.append({
                "Variable": col,
                "Percentile": f"P{int(p*100)}",
                "Empirical_Quantile": round(float(eq), 4),
                "Normal_Theoretical_Quantile": round(float(tq), 4),
                "Diff (Emp - Theo)": round(float(eq - tq), 4)
            })

    return pd.DataFrame(shape_rows), pd.DataFrame(quantile_rows)


def plot_normality_assessment(df_p2: pd.DataFrame, output_dir: str = "figures") -> str:
    """
    نمودار تسک ۶: مقایسه هیستوگرام + منحنی نرمال برازش‌شده و Q-Q Plot
    برای کادمیوم خام (ردیف بالا) در برابر کادمیوم لگاریتمی (ردیف پایین).
    """
    os.makedirs(output_dir, exist_ok=True)

    s_raw = df_p2["LBXBCD"].dropna().to_numpy(dtype=float)
    s_log = df_p2["Log10_LBXBCD"].dropna().to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ردیف ۱، چپ: هیستوگرام کادمیوم خام + منحنی نرمال
    mu_r, sd_r = np.mean(s_raw), np.std(s_raw, ddof=1)
    x_r = np.linspace(mu_r - 3.0 * sd_r, np.percentile(s_raw, 99.5), 400)
    axes[0, 0].hist(s_raw, bins=55, density=True, color="#e6550d", alpha=0.65, edgecolor="black", linewidth=0.4, label="Empirical Density")
    axes[0, 0].plot(x_r, stats.norm.pdf(x_r, mu_r, sd_r), color="black", linewidth=2.0, label=f"Normal PDF N({mu_r:.2f}, {sd_r:.2f}^2)")
    axes[0, 0].axvline(0.0, color="darkred", linestyle="--", linewidth=1.2, label="Physical Lower Bound (0 ug/L)")
    axes[0, 0].set_xlim(-1.0, 4.5)
    axes[0, 0].set_title("A. Raw Blood Cadmium vs. Fitted Normal Curve", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Blood Cadmium Concentration (ug/L)", fontsize=10)
    axes[0, 0].set_ylabel("Probability Density", fontsize=10)
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # ردیف ۱، راست: Q-Q Plot کادمیوم خام
    (osm_r, osr_r), (slope_r, intercept_r, _) = stats.probplot(s_raw, dist="norm")
    axes[0, 1].scatter(osm_r, osr_r, s=10, alpha=0.45, color="#e6550d", label="Sample Quantiles")
    axes[0, 1].plot(osm_r, slope_r * osm_r + intercept_r, color="black", linewidth=1.8, linestyle="--", label="Normal Reference Line")
    axes[0, 1].set_title("B. Normal Q-Q Plot: Raw Blood Cadmium (Severe Non-Normality)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Theoretical Standard Normal Quantiles (z)", fontsize=10)
    axes[0, 1].set_ylabel("Observed Raw Cadmium (ug/L)", fontsize=10)
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    # ردیف ۲، چپ: هیستوگرام کادمیوم لگاریتمی + منحنی نرمال
    mu_l, sd_l = np.mean(s_log), np.std(s_log, ddof=1)
    x_l = np.linspace(mu_l - 3.8 * sd_l, mu_l + 3.8 * sd_l, 400)
    axes[1, 0].hist(s_log, bins=40, density=True, color="#2b5c8f", alpha=0.68, edgecolor="black", linewidth=0.4, label="Empirical Density")
    axes[1, 0].plot(x_l, stats.norm.pdf(x_l, mu_l, sd_l), color="darkred", linewidth=2.0, label=f"Normal PDF N({mu_l:.2f}, {sd_l:.2f}^2)")
    axes[1, 0].set_title("C. Log10 Blood Cadmium vs. Fitted Normal Curve", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Log10 Blood Cadmium [log10(ug/L)]", fontsize=10)
    axes[1, 0].set_ylabel("Probability Density", fontsize=10)
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # ردیف ۲، راست: Q-Q Plot کادمیوم لگاریتمی
    (osm_l, osr_l), (slope_l, intercept_l, _) = stats.probplot(s_log, dist="norm")
    axes[1, 1].scatter(osm_l, osr_l, s=10, alpha=0.45, color="#2b5c8f", label="Sample Quantiles")
    axes[1, 1].plot(osm_l, slope_l * osm_l + intercept_l, color="darkred", linewidth=1.8, linestyle="--", label="Normal Reference Line")
    axes[1, 1].set_title("D. Normal Q-Q Plot: Log10 Blood Cadmium (Approximate Normality)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Theoretical Standard Normal Quantiles (z)", fontsize=10)
    axes[1, 1].set_ylabel("Observed Log10 Cadmium", fontsize=10)
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Assessment of Normal Approximation: Raw vs. Log10 Blood Cadmium (N = 4,655)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "p2_fig3_normality_cadmium.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path


# ==============================================================================
# تسک ۷: امید ریاضی و واریانس توزیع‌های تجربی گسسته و پیوسته
# ==============================================================================

def compute_expectation_and_variance(df_p2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    تسک ۷: محاسبه امید ریاضی (E[X])، واریانس (Var(X)) و انحراف معیار (SD(X))
    برای متغیرهای گسسته و پیوسته، به همراه جدول توزیع احتمال تجربی دقیق برای SMQ710.
    """
    # الف) جدول توزیع احتمال تجربی دقیق برای متغیر گسسته SMQ710 (تعداد روز مصرف سیگار در ۵ روز اخیر)
    s_days = df_p2["SMQ710"].dropna().astype(int)
    n_days = len(s_days)
    val_counts = s_days.value_counts().sort_index()

    pmf_rows = []
    mu_days = float(np.mean(s_days))
    for x_val, cnt in val_counts.items():
        px = cnt / n_days
        pmf_rows.append({
            "x (Smoking Days in Last 5d)": int(x_val),
            "Frequency (n)": int(cnt),
            "Empirical_Prob p(x)": round(px, 4),
            "x * p(x)": round(x_val * px, 4),
            "(x - E[X])^2 * p(x)": round(((x_val - mu_days) ** 2) * px, 4)
        })
    df_smq710_pmf = pd.DataFrame(pmf_rows)

    # ب) جدول جامع امید ریاضی و واریانس برای متغیرهای گسسته و پیوسته
    rv_specs = [
        ("Event_B_HighCd", "X_B: High Cadmium Indicator (Discrete Bernoulli)", "Discrete {0, 1}"),
        ("Event_C_ActiveCot", "X_C: Active Nicotine Indicator (Discrete Bernoulli)", "Discrete {0, 1}"),
        ("SMQ710", "D_cig: Smoking Days in Last 5d (Active Smokers)", "Discrete {1..5}"),
        ("LBXBCD", "W: Raw Blood Cadmium (ug/L)", "Continuous"),
        ("Log10_LBXBCD", "Y1: Log10 Blood Cadmium [log10(ug/L)]", "Continuous"),
        ("Log10_LBXCOT", "Y2: Log10 Serum Cotinine [log10(ng/mL)]", "Continuous"),
        ("LBXHGB", "Y3: Blood Hemoglobin (g/dL)", "Continuous")
    ]

    ev_rows = []
    for col, desc, rv_type in rv_specs:
        s = df_p2[col].dropna().to_numpy(dtype=float)
        mu = float(np.mean(s))
        var_emp = float(np.var(s, ddof=0))   # واریانس دقیق توزیع تجربی E[(X - mu)^2]
        var_samp = float(np.var(s, ddof=1))  # واریانس نمونه (با مخرج N-1)
        sd_emp = float(np.sqrt(var_emp))
        ev_rows.append({
            "RV_Name": desc,
            "Type": rv_type,
            "Valid_N": len(s),
            "E[X] (Mean)": round(mu, 4),
            "Var_Empirical E[(X-mu)^2]": round(var_emp, 4),
            "Var_Sample (N-1)": round(var_samp, 4),
            "SD(X)": round(sd_emp, 4)
        })

    # افزودن متغیر شمارشی K_30 بر پایه مدل دوجمله‌ای (np و np(1-p))
    p_b = float(df_p2["Event_B_HighCd"].mean())
    ev_rows.insert(3, {
        "RV_Name": "K_30: High-Cadmium Count in n=30 (Binomial Model)",
        "Type": "Discrete {0..30}",
        "Valid_N": len(df_p2),
        "E[X] (Mean)": round(30 * p_b, 4),
        "Var_Empirical E[(X-mu)^2]": round(30 * p_b * (1.0 - p_b), 4),
        "Var_Sample (N-1)": np.nan,
        "SD(X)": round(np.sqrt(30 * p_b * (1.0 - p_b)), 4)
    })

    df_ev_table = pd.DataFrame(ev_rows)
    return df_ev_table, df_smq710_pmf


# ==============================================================================
# تسک ۸: دو متغیر تصادفی و کوواریانس
# ==============================================================================

def compute_bivariate_covariance(df_p2: pd.DataFrame) -> pd.DataFrame:
    """
    تسک ۸: محاسبه کوواریانس تجربی Cov(X, Y)، جهت رابطه و سهم مشاهدات در ربع‌های هم‌جهت
    و خلاف‌جهت (مطابق فصل پنجم کتاب ووناکات، بدون ورود به همبستگی رسمی و آزمون فرض).
    """
    pairs = [
        ("Log10_LBXCOT", "Log10_LBXBCD", "Pair 1: Log10 Cotinine (Y2) & Log10 Cadmium (Y1)"),
        ("RIDAGEYR", "Log10_LBXBCD", "Pair 2: Age in Years (RIDAGEYR) & Log10 Cadmium (Y1)"),
        ("INDFMPIR", "Log10_LBXBCD", "Pair 3: Income-to-Poverty (INDFMPIR) & Log10 Cadmium (Y1)")
    ]

    rows = []
    for x_col, y_col, label in pairs:
        sub = df_p2[[x_col, y_col]].dropna()
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        mx, my = float(np.mean(x)), float(np.mean(y))
        dx, dy = x - mx, y - my

        cov_emp = float(np.mean(dx * dy))               # E[(X - mu_X)(Y - mu_Y)]
        cov_samp = float(np.cov(x, y, ddof=1)[0, 1])    # کوواریانس نمونه‌ای با مخرج N-1

        # سهم نقاط در ربع‌های ۱ و ۳ (حاصل‌ضرب مثبت) در برابر ربع‌های ۲ و ۴ (حاصل‌ضرب منفی)
        pct_pos_quad = float(np.mean((dx * dy) > 0) * 100)
        pct_neg_quad = float(np.mean((dx * dy) < 0) * 100)

        direction = "Positive (Direct)" if cov_samp > 0 else "Negative (Inverse)"

        rows.append({
            "Variable_Pair": label,
            "Valid_N": len(sub),
            "Mean_X": round(mx, 4),
            "Mean_Y": round(my, 4),
            "Cov_Empirical E[(X-mx)(Y-my)]": round(cov_emp, 4),
            "Cov_Sample (N-1)": round(cov_samp, 4),
            "Q1+Q3_Positive_Share (%)": round(pct_pos_quad, 2),
            "Q2+Q4_Negative_Share (%)": round(pct_neg_quad, 2),
            "Direction": direction
        })

    return pd.DataFrame(rows)


def plot_bivariate_covariance(df_p2: pd.DataFrame, output_dir: str = "figures") -> str:
    """
    نمودار تسک ۸: نمودار پراکنش سه زوج متغیر کمی همراه با خطوط میانگین ربع‌های کوواریانس
    (ربع‌های مثبت + و منفی - مطابق نمودار فصل پنجم کتاب ووناکات).
    """
    os.makedirs(output_dir, exist_ok=True)

    pair_specs = [
        ("Log10_LBXCOT", "Log10_LBXBCD", "Log10 Serum Cotinine [log10(ng/mL)]", "Log10 Blood Cadmium [log10(ug/L)]",
         "A. Nicotine Exposure vs. Blood Cadmium", "#2b5c8f"),
        ("RIDAGEYR", "Log10_LBXBCD", "Age (Years)", "Log10 Blood Cadmium [log10(ug/L)]",
         "B. Age vs. Blood Cadmium (Bioaccumulation)", "#d95f02"),
        ("INDFMPIR", "Log10_LBXBCD", "Income-to-Poverty Ratio (INDFMPIR)", "Log10 Blood Cadmium [log10(ug/L)]",
         "C. Income Ratio vs. Blood Cadmium (Inverse)", "#2ca02c")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    for ax, (x_col, y_col, xlabel, ylabel, title, color) in zip(axes, pair_specs):
        sub = df_p2[[x_col, y_col]].dropna()
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        mx, my = np.mean(x), np.mean(y)
        cov_val = np.cov(x, y, ddof=1)[0, 1]

        ax.scatter(x, y, s=11, alpha=0.25, color=color, edgecolors="none")
        ax.axvline(mx, color="black", linestyle="--", linewidth=1.3, label=f"Mean X = {mx:.2f}")
        ax.axhline(my, color="darkred", linestyle="--", linewidth=1.3, label=f"Mean Y = {my:.2f}")

        # نمایش علامت ربع‌های کوواریانس در گوشه‌های نمودار
        ax.text(0.95, 0.94, "Q1 (+)", transform=ax.transAxes, ha="right", va="top", fontsize=9, fontweight="bold", color="#1b5e20")
        ax.text(0.05, 0.06, "Q3 (+)", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color="#1b5e20")
        ax.text(0.05, 0.94, "Q2 (-)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color="#b71c1c")
        ax.text(0.95, 0.06, "Q4 (-)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold", color="#b71c1c")

        ax.set_title(f"{title}\nCov(X, Y) = {cov_val:+.4f}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle("Bivariate Scatter Plots and Covariance Quadrants (NHANES 2021-2023 Analytic Cohort)", fontsize=12.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "p2_fig4_bivariate_covariance.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    return out_path
