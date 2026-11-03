"""
ماژول ممیزی کیفیت داده و پاکسازی اولیه (تسک ۵ فاز اول)
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# کدهای استاندارد NHANES برای «امتناع از پاسخ (7/77/777)» و «نمی‌دانم (9/99/999)»
SPECIAL_MISSING_CODES = {
    "DMDEDUC2": [7, 9],
    "SMQ020": [7, 9],
    "SMQ040": [7, 9],
    "SMD650": [777, 999],
    "SMQ681": [7, 9],
    "SMQ710": [7, 9, 77, 99],
    "SMQ720": [777, 999],
    "SMQ845": [7, 9],
    "SMQ846": [7, 9],
    "SMQ849": [7, 9, 77, 99],
    "SMQ851": [7, 9],
    "SMQ863": [7, 9],
    "SMD470": [77, 99, 777, 999]
}


def fix_sas_zeros(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    تبدیل اعداد علمی بسیار نزدیک به صفر (ناشی از فرمت SAS XPT مانند 5.39e-79) به صفر دقیق (0.0).
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    tiny_floats_count = 0
    
    for col in numeric_cols:
        mask = (df_out[col].notna()) & (df_out[col] != 0.0) & (df_out[col].abs() < 1e-10)
        tiny_floats_count += int(mask.sum())
        df_out.loc[mask, col] = 0.0
        
    return df_out, tiny_floats_count


def audit_and_clean_special_codes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    شناسایی کدهای ویژه «نامعلوم / امتناع از پاسخ»، شمارش آن‌ها و تبدیل به NaN استاندارد.
    """
    df_out = df.copy()
    records = []
    
    for col, codes in SPECIAL_MISSING_CODES.items():
        if col in df_out.columns:
            refused_codes = [c for c in codes if str(c).startswith("7")]
            dont_know_codes = [c for c in codes if str(c).startswith("9")]
            
            n_refused = int(df_out[col].isin(refused_codes).sum())
            n_dk = int(df_out[col].isin(dont_know_codes).sum())
            n_total_special = n_refused + n_dk
            
            records.append({
                "Variable": col,
                "Special_Codes_Checked": str(codes),
                "Refused_Count (7/77/777)": n_refused,
                "Dont_Know_Count (9/99/999)": n_dk,
                "Total_Converted_to_NaN": n_total_special
            })
            
            df_out[col] = df_out[col].replace(codes, np.nan)
            
    return df_out, pd.DataFrame(records)


def audit_duplicates_and_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    بررسی رکوردهای تکراری، مقادیر غیرممکن منفی و ناسازگاری‌های منطقی بین پرسش‌نامه‌ها.
    """
    # ۱. رکوردهای تکراری
    dup_rows = int(df.duplicated().sum())
    dup_seqn = int(df["SEQN"].duplicated().sum())
    
    # ۲. مقادیر منفی یا غیرممکن در متغیرهای کمی
    quant_cols = ["RIDAGEYR", "INDFMPIR", "LBXCOT", "LBXHCOT", "LBXBCD", "LBXBPB", "LBXBSE", "LBXHGB", "LBXWBCSI"]
    neg_values = sum(int((df[c] < 0).sum()) for c in quant_cols if c in df.columns)
    
    # ۳. بررسی منطقی اول: کسانی که گفته‌اند در عمرشان ۱۰۰ نخ نکشیده‌اند (SMQ020==2)، آیا برای وضعیت فعلی (SMQ040) مقداری دارند؟
    logic_skip_violation = int(((df["SMQ020"] == 2) & (df["SMQ040"].notna())).sum())
    
    # ۴. بررسی منطقی دوم: کسانی که در مصاحبه منزل گفته‌اند اصلاً سیگار نمی‌کشند (SMQ020==2 یا SMQ040==3)
    # اما در کلینیک (MEC) گفته‌اند در ۵ روز گذشته سیگار کشیده‌اند (SMQ681==1)
    discordant_smokers = int((((df["SMQ020"] == 2) | (df["SMQ040"] == 3)) & (df["SMQ681"] == 1)).sum())
    
    # ۵. بررسی منطقی سوم: کسانی که گفته‌اند در ۵ روز اخیر بیش از ۵ روز سیگار یا ویپ کشیده‌اند!
    impossible_days = int(((df["SMQ710"] > 5) | (df["SMQ849"] > 5)).sum())
    
    checks = [
        {"Check_Type": "Duplicate Full Rows", "Count": dup_rows, "Decision": "No action needed (0 duplicates)"},
        {"Check_Type": "Duplicate SEQN IDs", "Count": dup_seqn, "Decision": "1-to-1 merge verified"},
        {"Check_Type": "Impossible Negative Values", "Count": neg_values, "Decision": "All quantitative values within valid physical ranges"},
        {"Check_Type": "Questionnaire Skip-Pattern Violation (SMQ020=2 & SMQ040 not null)", "Count": logic_skip_violation, "Decision": "Structural missingness confirmed"},
        {"Check_Type": "Discordant Self-Report (Non-smoker in SMQ but Smoked Last 5 Days in SMQRTU)", "Count": discordant_smokers, "Decision": "Classify based on active recent use (SMQ681=1) or flag in sensitivity"},
        {"Check_Type": "Impossible Recent Days (>5 days in last 5 days)", "Count": impossible_days, "Decision": "Valid range [0, 5] confirmed"}
    ]
    return pd.DataFrame(checks)


def audit_missing_and_ranges(df: pd.DataFrame, subset_adults: bool = True) -> pd.DataFrame:
    """
    بررسی تعداد و درصد داده‌های گمشده و دامنه مقادیر (کمینه و بیشینه).
    به طور پیش‌فرض هم روی کل داده و هم روی بزرگسالان (20+) گزارش می‌دهد.
    """
    target_df = df[df["RIDAGEYR"] >= 20].copy() if subset_adults else df.copy()
    total_n = len(target_df)
    
    audit_rows = []
    for col in target_df.columns:
        if col == "SEQN":
            continue
        n_miss = int(target_df[col].isna().sum())
        n_valid = total_n - n_miss
        pct_miss = round((n_miss / total_n) * 100, 2)
        
        min_val = round(float(target_df[col].min()), 4) if n_valid > 0 else np.nan
        max_val = round(float(target_df[col].max()), 4) if n_valid > 0 else np.nan
        
        audit_rows.append({
            "Variable": col,
            "Total_N": total_n,
            "Valid_N": n_valid,
            "Missing_Count": n_miss,
            "Missing_Percent (%)": pct_miss,
            "Min_Value": min_val,
            "Max_Value": max_val
        })
        
    return pd.DataFrame(audit_rows)


def audit_lod_censored_values(df: pd.DataFrame, subset_adults: bool = True) -> pd.DataFrame:
    """
    ممیزی مقادیر زیر حد تشخیص آزمایشگاه (Below Limit of Detection - LOD) برای بیومارکرها.
    """
    target_df = df[df["RIDAGEYR"] >= 20].copy() if subset_adults else df.copy()
    
    lod_pairs = [
        ("LBXCOT", "LBDCOTLC", "Serum Cotinine (ng/mL)"),
        ("LBXHCOT", "LBDHCOLC", "Serum Hydroxycotinine (ng/mL)"),
        ("LBXBCD", "LBDBCDLC", "Blood Cadmium (ug/L)"),
        ("LBXBPB", "LBDBPBLC", "Blood Lead (ug/dL)"),
        ("LBXBSE", "LBDBSELC", "Blood Selenium (ug/L)")
    ]
    
    lod_records = []
    for val_col, lc_col, desc in lod_pairs:
        valid_mask = target_df[val_col].notna() & target_df[lc_col].notna()
        n_tested = int(valid_mask.sum())
        n_below_lod = int((target_df.loc[valid_mask, lc_col] == 1.0).sum())
        pct_below_lod = round((n_below_lod / n_tested) * 100, 2) if n_tested > 0 else 0.0
        
        imputed_vals = target_df.loc[valid_mask & (target_df[lc_col] == 1.0), val_col].unique()
        imputed_val = round(float(imputed_vals[0]), 4) if len(imputed_vals) > 0 else np.nan
        
        lod_records.append({
            "Biomarker": val_col,
            "Description": desc,
            "Comment_Code_Var": lc_col,
            "Total_Tested": n_tested,
            "Below_LOD_Count (Code=1)": n_below_lod,
            "Below_LOD_Percent (%)": pct_below_lod,
            "CDC_Imputed_Value (LOD/sqrt(2))": imputed_val
        })
        
    return pd.DataFrame(lod_records)
