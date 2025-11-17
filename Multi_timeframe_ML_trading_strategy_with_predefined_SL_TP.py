"""
ALGOTRADING PIPELINE - EURUSD PREDICTION
Author: [sklepfuzja]
Date: 2024

Description: Multi-timeframe ensemble system with meta-learning for EURUSD forecasting.

Features:
• Multi-timeframe data with offset sampling (M1-M6)
• Ensemble modeling (XGBoost + Logistic Regression)  
• Two-stage meta-learning with prediction stacking
• Trading strategy with dynamic Stop-Loss/Take-Profit
• Walk-forward validation with risk management

Pipeline:
Data → Multi-timeframe Features → Base Models → Meta Ensemble → Trading Strategy → Evaluation
"""

# ==================== IMPORTS ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer

# Data & Features
import MetaTrader5 as mt5
from Data_download import DataFetcherMT5, PrepareData
from Utility_2 import MultiTargetTransformer, StationaryOutlierRemover

# Additional
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

# Config
from sklearn import set_config
set_config(transform_output="pandas")
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==================== CONFIGURATION ====================
SYMBOL = 'EURUSD'
TIMEFRAME = 'M1'
DATE_FROM = datetime(2025, 1, 28)
DATE_TO = datetime(2025, 2, 1)

# MT5 Timeframes
M1, M2, M3, M4, M5, M6, M10, M15, M30 = [
    mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M2, mt5.TIMEFRAME_M3,
    mt5.TIMEFRAME_M4, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M6,
    mt5.TIMEFRAME_M10, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30
]

H1, H2, H3, H4, H12, D1, W1, MN1 = [
    mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H2, mt5.TIMEFRAME_H3,
    mt5.TIMEFRAME_H4, mt5.TIMEFRAME_H12, mt5.TIMEFRAME_D1,
    mt5.TIMEFRAME_W1, mt5.TIMEFRAME_MN1
]

# Trading parameters
SPREAD = 0.00001
COMMISSION = 0.00005
TRADING_HOURS = {'start': '02:00', 'end': '22:00'}

# ==================== DATA FETCHING ====================
print("📊 Fetching data...")

# MT5 Login
fetcher = DataFetcherMT5(
    login=None, 
    password=None, 
    server=None
)

# Fetch tick data
df_ticks = fetcher.fetch_data_ticks_range(
    symbol=SYMBOL, 
    date_from=DATE_FROM, 
    date_to=DATE_TO
)

print(f"✅ Data fetched - ticks: {df_ticks.shape}")

# ==================== DATA AGGREGATION ====================
print("🔄 Aggregating multi-timeframe data...")

# Create multiple timeframe datasets
timeframe_configs = [
    ('1T', None, 'df1m'),
    ('2T', None, 'df2m0'), ('2T', '1T', 'df2m1'),
    ('3T', None, 'df3m0'), ('3T', '1T', 'df3m1'), ('3T', '2T', 'df3m2'),
    ('4T', None, 'df4m0'), ('4T', '1T', 'df4m1'), ('4T', '2T', 'df4m2'), ('4T', '3T', 'df4m3'),
    ('5T', None, 'df5m0'), ('5T', '1T', 'df5m1'), ('5T', '2T', 'df5m2'), 
    ('5T', '3T', 'df5m3'), ('5T', '4T', 'df5m4'),
    ('6T', None, 'df6m0'), ('6T', '1T', 'df6m1'), ('6T', '2T', 'df6m2'),
    ('6T', '3T', 'df6m3'), ('6T', '4T', 'df6m4'), ('6T', '5T', 'df6m5')
]

# Store all dataframes in a dictionary
dataframes = {}

for freq, offset, name in timeframe_configs:
    dataframes[name] = fetcher.aggregate_data_bid(
        frequency=freq, 
        offset=offset, 
        df=df_ticks
    )

print("✅ Multi-timeframe aggregation completed")

# ==================== FEATURE ENGINEERING ====================
print("⚙️ Computing features...")

feature_engineer = PrepareData()
start_time = time.time()

# Process all dataframes with feature engineering
for name in dataframes.keys():
    dataframes[name] = feature_engineer.feature_dataset_1(dataframes[name])

print(f"✅ Features computed in {time.time() - start_time:.2f}s")

# ==================== DATA COMBINATION ====================
print("🔗 Combining timeframe datasets...")

# Combine offset datasets
dataframes['df2m'] = pd.concat([dataframes['df2m0'], dataframes['df2m1']], axis=0).sort_index().fillna(0)
dataframes['df3m'] = pd.concat([dataframes['df3m0'], dataframes['df3m1'], dataframes['df3m2']], axis=0).sort_index().fillna(0)
dataframes['df4m'] = pd.concat([dataframes['df4m0'], dataframes['df4m1'], dataframes['df4m2'], dataframes['df4m3']], axis=0).sort_index().fillna(0)
dataframes['df5m'] = pd.concat([dataframes['df5m0'], dataframes['df5m1'], dataframes['df5m2'], dataframes['df5m3'], dataframes['df5m4']], axis=0).sort_index().fillna(0)
dataframes['df6m'] = pd.concat([dataframes['df6m0'], dataframes['df6m1'], dataframes['df6m2'], dataframes['df6m3'], dataframes['df6m4'], dataframes['df6m5']], axis=0).sort_index().fillna(0)

print("✅ Timeframe combination completed")

# ==================== TARGET CREATION ====================
print("🎯 Creating targets...")

def create_all_targets(df, shift, target_type='binary'):
    """Create all target types for a given dataframe."""
    return feature_engineer.create_targets_with_specific_shift(
        df, 
        type=target_type, 
        validation_dataset=True, 
        shift=shift,
        start_time=TRADING_HOURS['start'],
        end_time=TRADING_HOURS['end'],
        shuffle_after_split=False
    )

# Create targets for all timeframes
targets = {}
timeframe_shifts = {
    'df1m': 1, 'df2m': 2, 'df3m': 3, 
    'df4m': 4, 'df5m': 5, 'df6m': 6
}

for df_name, shift in timeframe_shifts.items():
    # Binary targets
    targets[f'X_train_{df_name}'], targets[f'X_val_{df_name}'], targets[f'X_test_{df_name}'], \
    targets[f'y_train_{df_name}'], targets[f'y_val_{df_name}'], targets[f'y_test_{df_name}'] = create_all_targets(
        dataframes[df_name], shift, 'binary'
    )
    
    # Regression diff targets
    _, _, _, targets[f'y_train_{df_name}_reg_diff'], targets[f'y_val_{df_name}_reg_diff'], targets[f'y_test_{df_name}_reg_diff'] = create_all_targets(
        dataframes[df_name], shift, 'reg_diff'
    )
    
    # Regression raw targets  
    _, _, _, targets[f'y_train_{df_name}_reg_raw'], targets[f'y_val_{df_name}_reg_raw'], targets[f'y_test_{df_name}_reg_raw'] = create_all_targets(
        dataframes[df_name], shift, 'reg_raw'
    )

print("✅ Target creation completed")

# ==================== BASE MODEL TRAINING ====================
print("🤖 Training base models...")

# Multi-target pipeline configuration
base_pipeline = Pipeline(steps=[
    ('simple_imputer', SimpleImputer()),
    ('scaler', MinMaxScaler()),
    ('pca', PCA()),
    ('classifier2', LogisticRegression())
])

multi_target_transformer = MultiTargetTransformer(
    base_pipeline=base_pipeline, 
    n_targets=targets['y_train_df1m'].shape[1]
)

# Train base models for all timeframes
base_predictions = {}

for df_name in ['df1m', 'df2m', 'df3m', 'df4m', 'df5m', 'df6m']:
    X_train = targets[f'X_train_{df_name}']
    y_train = targets[f'y_train_{df_name}']
    X_val = targets[f'X_val_{df_name}']
    X_test = targets[f'X_test_{df_name}']
    
    multi_target_transformer.fit(X_train, y_train)
    
    base_predictions[f'train_{df_name}'] = multi_target_transformer.predict(X_train)
    base_predictions[f'val_{df_name}'] = multi_target_transformer.predict(X_val)
    base_predictions[f'test_{df_name}'] = multi_target_transformer.predict(X_test)

print("✅ Base models trained")

# ==================== PREDICTION AGGREGATION ====================
print("🔄 Aggregating predictions...")

def create_prediction_dataframe(prediction_dict, prefix):
    """Create unified dataframe from multiple predictions."""
    common_index = None
    
    # Find common index
    for key in prediction_dict.keys():
        if prefix in key:
            if common_index is None:
                common_index = prediction_dict[key].index
            else:
                common_index = common_index.union(prediction_dict[key].index)
    
    # Create unified dataframe
    result_df = pd.DataFrame(index=common_index)
    
    for key, predictions in prediction_dict.items():
        if prefix in key:
            timeframe_suffix = key.replace(prefix, '').upper()
            result_df = result_df.join(
                predictions.add_suffix(f'_{timeframe_suffix}'), 
                how='left'
            )
    
    return result_df

# Create unified prediction datasets
train_predictions = create_prediction_dataframe(base_predictions, 'train_')
val_predictions = create_prediction_dataframe(base_predictions, 'val_')
test_predictions = create_prediction_dataframe(base_predictions, 'test_')

print("✅ Prediction aggregation completed")

# ==================== ENHANCED FEATURE CREATION ====================
print("✨ Creating enhanced features...")

# Combine original features with predictions
enhanced_features = {}

for df_name in ['df1m', 'df2m', 'df3m', 'df4m', 'df5m', 'df6m']:
    enhanced_features[f'train_{df_name}'] = pd.concat([
        targets[f'X_train_{df_name}'], 
        train_predictions
    ], axis=1)
    
    enhanced_features[f'val_{df_name}'] = pd.concat([
        targets[f'X_val_{df_name}'], 
        val_predictions
    ], axis=1)
    
    enhanced_features[f'test_{df_name}'] = pd.concat([
        targets[f'X_test_{df_name}'], 
        test_predictions
    ], axis=1)

print("✅ Enhanced features created")

# ==================== ALIGN DATA INDICES ====================
print("🔧 Aligning data indices...")

def align_data_indices(features_df, targets_dict, prefix):
    """Align features and targets by index."""
    aligned_data = {}
    
    for df_name in ['df1m', 'df2m', 'df3m', 'df4m', 'df5m', 'df6m']:
        features_key = f'{prefix}_{df_name}'
        targets_key = f'y_{prefix}_{df_name}'
        
        if features_key in features_df and targets_key in targets_dict:
            common_index = features_df[features_key].index.intersection(
                targets_dict[targets_key].index
            )
            
            aligned_data[features_key] = features_df[features_key].loc[common_index]
            aligned_data[targets_key] = targets_dict[targets_key].loc[common_index]
            aligned_data[f'{targets_key}_reg_diff'] = targets_dict[f'{targets_key}_reg_diff'].loc[common_index]
            aligned_data[f'{targets_key}_reg_raw'] = targets_dict[f'{targets_key}_reg_raw'].loc[common_index]
    
    return aligned_data

# Align all datasets
aligned_train = align_data_indices(enhanced_features, targets, 'train')
aligned_val = align_data_indices(enhanced_features, targets, 'val')  
aligned_test = align_data_indices(enhanced_features, targets, 'test')

print("✅ Data alignment completed")

# ==================== META MODEL TRAINING ====================
print("🧠 Training meta models...")

meta_predictions = {}

for df_name in ['df1m', 'df2m', 'df3m', 'df4m', 'df5m', 'df6m']:
    X_train = aligned_train[f'train_{df_name}']
    y_train = aligned_train[f'y_train_{df_name}']
    X_val = aligned_val[f'val_{df_name}']
    X_test = aligned_test[f'test_{df_name}']
    
    # Proste wypełnianie NaN dla metalearning też
    X_train_clean = X_train.fillna(0)
    X_val_clean = X_val.fillna(0)
    X_test_clean = X_test.fillna(0)
    
    multi_target_transformer.fit(X_train_clean, y_train)
    
    meta_predictions[f'val_{df_name}'] = multi_target_transformer.predict(X_val_clean).add_suffix(f'_{df_name.upper()}')
    meta_predictions[f'test_{df_name}'] = multi_target_transformer.predict(X_test_clean).add_suffix(f'_{df_name.upper()}')

print("✅ Meta models trained")

# ==================== FINAL ENSEMBLE ====================
print("🎯 Creating final ensemble...")

# Combine all meta predictions
final_X_val = pd.concat([
    targets['X_val_df1m']
] + [meta_predictions[f'val_{df_name}'] for df_name in ['df1m', 'df2m', 'df3m', 'df4m', 'df5m', 'df6m']], axis=1).fillna(0)

final_X_test = pd.concat([
    targets['X_test_df1m']  
] + [meta_predictions[f'test_{df_name}'] for df_name in ['df1m', 'df2m', 'df3m', 'df4m', 'df5m', 'df6m']], axis=1).fillna(0)

print(f"✅ Final ensemble created - Validation: {final_X_val.shape}, Test: {final_X_test.shape}")

# ==================== TRADING STRATEGY ====================
print("💹 Setting up trading strategy...")

def calculate_trading_targets(df, spread=SPREAD, commission=COMMISSION):
    """Calculate stop loss, take profit and trading targets."""
    df = df.copy()
    
    # Calculate SL and TP for sell positions
    df['SL'] = np.where(df['high'] > df['high'].shift(1), df['high'], df['high'].shift(1)) + spread + commission
    df['risk'] = df['SL'] - df['close']
    df['TP'] = df['close'] - 1 * df['risk'] - spread - commission
    
    def calculate_target(row_index, dataframe):
        """Calculate if trade would be successful."""
        current_row = dataframe.iloc[row_index]
        
        for i in range(1, 6):  # Check next 5 candles
            if row_index + i >= len(dataframe):
                return 0
                
            future_row = dataframe.iloc[row_index + i]
            
            if future_row['high'] >= current_row['SL']:  # Stop loss hit
                return 0
            if future_row['low'] <= current_row['TP']:  # Take profit hit  
                return 1
                
        return 0
    
    # Apply target calculation
    df['target'] = [calculate_target(ix, df) for ix in range(len(df))]
    df = df.fillna(0)
    
    return df

# Apply trading strategy to validation data
trading_df = calculate_trading_targets(final_X_val)

print(f"📊 Target distribution:\n{trading_df['target'].value_counts()}")
print(f"📈 Average risk for successful trades: {trading_df[trading_df['target'] == 1]['risk'].mean():.6f}")

# ==================== FINAL MODEL EVALUATION ====================
print("🔍 Running final model evaluation...")

# Training parameters
TRAIN_SAMPLES = 800
TEST_SAMPLES = 200
total_samples = len(trading_df)

# Final pipeline
final_pipeline = Pipeline([
    ('simple_imputer', SimpleImputer()),
    ('calibrated', CalibratedClassifierCV(XGBClassifier(scale_pos_weight=1), cv=3)),
])

# Walk-forward validation
results = []
all_predictions = []
all_actuals = []

for start_idx in range(0, total_samples - TRAIN_SAMPLES, TEST_SAMPLES):
    end_train = start_idx + TRAIN_SAMPLES
    end_test = min(end_train + TEST_SAMPLES, total_samples)
    
    # Prepare datasets
    train_data = trading_df.iloc[start_idx:end_train]
    test_data = trading_df.iloc[end_train:end_test]
    
    X_train, y_train = train_data.drop('target', axis=1), train_data['target']
    X_test, y_test = test_data.drop('target', axis=1), test_data['target']
    
    # Train and predict
    final_pipeline.fit(X_train, y_train)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Store results
    accuracy = accuracy_score(y_test, y_pred)
    results.append(accuracy)
    
    all_predictions.extend(y_pred)
    all_actuals.extend(y_test)

# Calculate metrics
average_accuracy = np.mean(results)
y_pred_final = np.array(all_predictions)
y_test_final = np.array(all_actuals)

TP = np.sum((y_pred_final == 1) & (y_test_final == 1))
FP = np.sum((y_pred_final == 1) & (y_test_final == 0))

if TP + FP > 0:
    effectiveness = (TP / (TP + FP)) * 100
else:
    effectiveness = 0

print(f"📈 Average Accuracy: {average_accuracy:.2%}")
print(f"🎯 Trading Effectiveness: {effectiveness:.2f}%")
print(f"✅ True Positives: {TP}")
print(f"❌ False Positives: {FP}")

# ==================== VISUALIZATION ====================
print("📈 Generating visualizations...")

# Simple plots
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(y_test_final, label='Actual', alpha=0.7)
plt.plot(y_pred_final, label='Predicted', alpha=0.7)
plt.title('Predictions vs Actual')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results)
plt.title('Accuracy Over Iterations')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

print("🎉 Pipeline completed successfully!")