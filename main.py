import pandas as pd
import numpy as np
import zipfile
import os
import glob
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ---------- 1. data loading ----------
def extract_all_zips():
    for zip_path in DATA_DIR.glob("*.zip"):
        print(f"extracting: {zip_path.name}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_DIR / "extracted")
    print("extraction done")

def parse_median(text):
    """Turn '30 min' or '1 hour' to minutes(int)"""
    if pd.isna(text):
        return np.nan
    text = str(text)
    nums = re.findall(r'\d+', text)
    if not nums:
        return np.nan
    if 'hour' in text.lower():
        return int(nums[0]) * 60
    else:
        return int(nums[0])

def load_all_excel():
    extracted_dir = DATA_DIR / "extracted"
    excel_files = glob.glob(str(extracted_dir / "**" / "*.xlsx"), recursive=True)
    print(f"found {len(excel_files)} excel files")
    
    all_dfs = []
    for fp in excel_files:  
        try:
            base = os.path.basename(fp)
            yyyymmdd = base.split('-')[0]
            hhmm = base.split('-')[1]
            timestamp = pd.to_datetime(f"{yyyymmdd} {hhmm[:2]}:{hhmm[2:]}", 
                                       format='%Y%m%d %H:%M')
            df = pd.read_excel(fp, header=0, skiprows=8, usecols='A:G')
            df.columns = ['hospital', 'wait_I', 'treating_I', 
                          'wait_II', 'treating_II', 'wait_III', 'wait_IV_V']
            df['timestamp'] = timestamp
            all_dfs.append(df)
        except Exception as e:
            print(f"Wrong file: {fp} -> {e}")
    
    return pd.concat(all_dfs, ignore_index=True)

# ---------- 2. feature engineering ----------
def engineer_features(df):
    df = df.copy()
    
    # 2.1 swap wait_IV_V to median minutes
    df['wait_IV_V_med'] = df['wait_IV_V'].apply(parse_median)
    
    # 2.2 handle treating_I and treating_II: map 'Y' to 1, 'N' to 0, NaN to 0
    df['treating_I'] = df['treating_I'].map({'Y':1,'N':0}).fillna(0)
    df['treating_II'] = df['treating_II'].map({'Y':1,'N':0}).fillna(0)
    
    # 2.3 overload indicator: if treating_I or treating_II is NaN, set resus_overload=1
    df['resus_overload'] = df['treating_I'].isna().astype(int) | df['treating_II'].isna().astype(int)
    df['treating_I'] = df['treating_I'].fillna(0)
    df['treating_II'] = df['treating_II'].fillna(0)
    
    # 2.4 cut off wait_IV_V_med at 720 mins (12 hours)
    df['wait_IV_V_med'] = df['wait_IV_V_med'].clip(upper=720)
    
    # 2.5 compute lag features and rolling mean for each hospital
    df = df.sort_values(['hospital', 'timestamp'])
    for h in df['hospital'].unique():
        mask = df['hospital'] == h
        df.loc[mask, 'lag1'] = df.loc[mask, 'wait_IV_V_med'].shift(1)      # 15分钟前
        df.loc[mask, 'lag4'] = df.loc[mask, 'wait_IV_V_med'].shift(4)      # 1小时前
        df.loc[mask, 'roll_mean4'] = df.loc[mask, 'wait_IV_V_med'].rolling(4, min_periods=1).mean()
    
    # 2.6 timestamp -> hour, day_of_week, is_weekend
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 2.7 target variable: 30 mins later, is the hospital crowded? (wait_IV_V_med > 120)
    df['crowded'] = (df['wait_IV_V_med'] > 120).astype(int)
    df['target'] = df.groupby('hospital')['crowded'].shift(-2)
    
    # delete rows where target is NaN (i.e., the last two rows for each hospital)
    df = df.dropna(subset=['target'])
    
    return df


def train_and_evaluate(df):
    train_df = df[df['timestamp'] < '2026-03-01']
    test_df = df[df['timestamp'] >= '2026-03-01']
    
    features = ['lag1', 'lag4', 'roll_mean4', 'treating_I', 'treating_II', 
                'resus_overload', 'hour', 'day_of_week', 'is_weekend']
    
    X_train = train_df[features].fillna(0)
    y_train = train_df['target']
    X_test = test_df[features].fillna(0)
    y_test = test_df['target']
    
    print(f"training: {len(X_train)}, testing: {len(X_test)}")
    print(f"ratio of crowded cases: {y_train.mean():.3f}")
    
    scale = (y_train == 0).sum() / (y_train == 1).sum() if y_train.sum() > 0 else 1
    
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(scale_pos_weight=scale, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        results[name] = {
            'AUC_ROC': roc_auc_score(y_test, y_prob),
            'PR_AUC': average_precision_score(y_test, y_prob),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred)
        }
    
    result_df = pd.DataFrame(results).T.round(4)
    print("\nmodel results:")
    print(result_df)
    
    # XGBoost
    xgb_model = models['XGBoost']
    importance = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(8, 4))
    plt.barh(importance['feature'], importance['importance'])
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')  # 保存到当前文件夹
    print("\nFeature importance plot saved as feature_importance.png")
    
    return result_df

if __name__ == "__main__":
    extract_all_zips()

    print("\nloading data...")
    raw_df = load_all_excel()

    print("\nfeature engineering...")
    model_df = engineer_features(raw_df)
    
    print("\ntraining and evaluating...")
    results = train_and_evaluate(model_df)
    
    
    print(f"\nfinal data shape: {model_df.shape}")
    print(f"crowded ratio (target=1): {model_df['target'].mean():.3f}")
    print("\nfeature columns preview:")
    print(model_df[['hospital', 'timestamp', 'wait_IV_V_med', 'lag1', 'lag4', 'roll_mean4', 'target']].head())
    model_df.to_csv('hospital_forecast_data.csv', index=False)
    print("hospital_forecast_data.csv saved.")