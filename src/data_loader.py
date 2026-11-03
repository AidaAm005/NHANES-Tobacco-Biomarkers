"""
ماژول بارگذاری و ادغام فایل‌های خام NHANES 2021-2023 (تسک الف-۱)
"""
import pandas as pd
from typing import Dict, Tuple

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"

# دیکشنری فایل‌های مورد استفاده و ستون‌های منتخب هر فایل طبق کارت پروژه
SELECTED_COLUMNS = {
    "DEMO_L": [
        "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", 
        "INDFMPIR", "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"
    ],
    "SMQ_L": ["SEQN", "SMQ020", "SMQ040", "SMD650"],
    "SMQRTU_L": [
        "SEQN", "SMQ681", "SMQ710", "SMQ720", "SMQ845", 
        "SMQ846", "SMQ849", "SMQ851", "SMQ863"
    ],
    "SMQFAM_L": ["SEQN", "SMD470"],
    "COT_L": ["SEQN", "LBXCOT", "LBDCOTLC", "LBXHCOT", "LBDHCOLC"],
    "PBCD_L": [
        "SEQN", "WTPH2YR", "LBXBCD", "LBDBCDLC", 
        "LBXBPB", "LBDBPBLC", "LBXBSE", "LBDBSELC"
    ],
    "CBC_L": ["SEQN", "LBXHGB", "LBXWBCSI"]
}


def fetch_raw_datasets() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    فایل‌های XPT را از سرور رسمی CDC دانلود کرده و جدول ابعاد اولیه (قبل از ادغام) را برمی‌گرداند.
    """
    datasets = {}
    summary_records = []

    for file_name, cols in SELECTED_COLUMNS.items():
        url = f"{BASE_URL}{file_name}.xpt"
        df_raw = pd.read_sas(url, format="xport")
        df_raw["SEQN"] = df_raw["SEQN"].astype(int)
        
        summary_records.append({
            "File_Name": file_name,
            "Total_Rows_Raw": len(df_raw),
            "Unique_SEQN_Raw": df_raw["SEQN"].nunique(),
            "Total_Columns_Raw": df_raw.shape[1],
            "Selected_Columns_Count": len(cols)
        })
        
        # نگه‌داشتن تنها ستون‌های مورد نیاز پروژه برای بهینه‌سازی حافظه
        datasets[file_name] = df_raw[cols].copy()

    df_pre_merge_summary = pd.DataFrame(summary_records)
    return datasets, df_pre_merge_summary


def merge_datasets(datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ادغام فایل‌ها بر اساس کلید مشترک SEQN به دو روش:
    1. Left Join با محوریت DEMO_L (حفظ کل چارچوب نمونه)
    2. بررسی اشتراک کامل (Inner Join) برای گزارش تعداد افراد مشترک در تمام فایل‌ها
    """
    df_merged = datasets["DEMO_L"].copy()
    
    merge_steps = [{
        "Merge_Step": "1. Base File: DEMO_L",
        "Merged_File": "DEMO_L",
        "Merge_Key": "SEQN",
        "Total_Rows_Left_Join": len(df_merged),
        "Matched_Participants_in_Step": len(df_merged),
        "Cumulative_Complete_Overlap (Inner)": len(df_merged)
    }]
    
    df_inner_tracker = datasets["DEMO_L"][["SEQN"]].copy()
    
    merge_order = ["SMQ_L", "SMQRTU_L", "SMQFAM_L", "COT_L", "PBCD_L", "CBC_L"]
    
    for idx, file_name in enumerate(merge_order, start=2):
        df_right = datasets[file_name].copy()
        # ایجاد ستون پرچم موقت برای شمارش دقیق افرادی که در فایل جدید تطابق دارند
        df_right["_matched_flag"] = 1
        
        df_merged = pd.merge(df_merged, df_right, on="SEQN", how="left")
        matched_count = int(df_merged["_matched_flag"].sum())
        df_merged.drop(columns=["_matched_flag"], inplace=True)
        
        # پیگیری تعداد افراد در صورت ادغام اشتراکی (Inner Join)
        df_inner_tracker = pd.merge(df_inner_tracker, datasets[file_name][["SEQN"]], on="SEQN", how="inner")
        
        merge_steps.append({
            "Merge_Step": f"{idx}. Merge {file_name}",
            "Merged_File": file_name,
            "Merge_Key": "SEQN",
            "Total_Rows_Left_Join": len(df_merged),
            "Matched_Participants_in_Step": matched_count,
            "Cumulative_Complete_Overlap (Inner)": len(df_inner_tracker)
        })
        
    df_merge_report = pd.DataFrame(merge_steps)
    return df_merged, df_merge_report
