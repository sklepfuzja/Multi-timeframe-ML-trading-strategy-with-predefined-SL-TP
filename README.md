# Multi-timeframe ML Trading Strategy with Predefined SL/TP

Advanced ensemble machine learning system for EURUSD forecasting with dynamic risk management.

## Features

- **Multi-timeframe Data**: Offset sampling across M1-M6 timeframes
- **Ensemble Modeling**: XGBoost + Logistic Regression combination
- **Two-stage Meta-learning**: Prediction stacking for enhanced performance
- **Dynamic Risk Management**: Automated Stop-Loss/Take-Profit calculation
- **Walk-forward Validation**: Robust out-of-sample testing methodology

## Pipeline Architecture

Data → Multi-timeframe Features → Base Models → Meta Ensemble → Trading Strategy → Evaluation

## Technical Implementation

### Data Processing
- MT5 tick data aggregation with multiple offset configurations
- Advanced feature engineering across all timeframes
- Stationarity transformation and outlier removal
- Multi-target label creation (binary & regression)

### Machine Learning
- **Base Models**: Multi-target transformer with PCA feature reduction
- **Meta Models**: Enhanced feature space with prediction stacking
- **Final Ensemble**: Combined predictions from all timeframes
- **Calibration**: Probability calibration for reliable trading signals

### Trading Strategy
- Dynamic SL/TP based on recent price volatility
- Spread and commission-aware position sizing
- Trading hours filtering (02:00-22:00)
- Risk-adjusted target calculation

## Performance Metrics

- **Average Accuracy**: Model prediction accuracy across validation folds
- **Trading Effectiveness**: Percentage of profitable trades among signals
- **Risk Assessment**: Average risk per successful trade

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Disclaimer
This trading system is for educational and research purposes. Always test strategies thoroughly with historical data and paper trading before deploying with real capital. Past performance does not guarantee future results.
