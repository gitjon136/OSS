import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import platform
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# --- 1~3. 데이터 준비 (이전과 동일) ---
START_DATE = '2020-01-01'
END_DATE = '2025-10-08'
# (데이터 준비 코드는 변경 없음)
# ... (이전과 동일한 데이터 준비 코드를 여기에 붙여넣으세요) ...
YFINANCE_MAP = {
    '^KS11': 'KOSPI', '^GSPC': 'S&P500', '^IXIC': 'NASDAQ',
    'KRW=X': 'USD_KRW', 'CL=F': 'WTI_OIL', 'GC=F': 'GOLD'
}
FRED_TICKERS = {
    'DGS10': 'US_10Y_TREASURY', 'DGS1': 'US_1Y_TREASURY', 'DGS3MO': 'US_3M_TREASURY',
    'IRLTLT01KRM156N': 'KOR_10Y_TREASURY', 'IR3TIB01KRM156N': 'KOR_3M_TREASURY',
    'CPIAUCSL': 'US_CPI', 'KORCPIALLMINMEI': 'KOR_CPI',
    'UNRATE': 'US_Unemployment', 'LRUNTTTTKRM156S': 'KOR_Unemployment',
    'UMCSENT': 'US_CSI'
}
print("데이터를 수집하고 전처리합니다...")
df_market = yf.download(list(YFINANCE_MAP.keys()), start=START_DATE, end=END_DATE, progress=False, timeout=15)
df_market = df_market['Close']
df_market.rename(columns=YFINANCE_MAP, inplace=True)
df_econ = web.DataReader(list(FRED_TICKERS.keys()), 'fred', START_DATE, END_DATE)
df_econ.columns = list(FRED_TICKERS.values())
try:
    df_kor_1y = pd.read_excel('KOR_1Y_TREASURY.xlsx', index_col=0, parse_dates=True)
    df_kor_1y.columns = ['KOR_1Y_TREASURY']
except FileNotFoundError:
    df_kor_1y = pd.DataFrame()
df = df_market.merge(df_econ, left_index=True, right_index=True, how='left')
if not df_kor_1y.empty:
    df = df.merge(df_kor_1y, left_index=True, right_index=True, how='left')
df.index.name = 'DATE'
df['US_Yield_Curve'] = df['US_10Y_TREASURY'] - df['US_3M_TREASURY']
df['KOR_Yield_Curve'] = df['KOR_10Y_TREASURY'] - df['KOR_3M_TREASURY']
df.ffill(inplace=True); df.bfill(inplace=True)
for col in df.columns:
    df[f'{col}_MA5'] = df[col].rolling(window=5).mean()
    df[f'{col}_MA20'] = df[col].rolling(window=20).mean()
    df[f'{col}_Momentum'] = df[col].pct_change()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True); df.bfill(inplace=True)
print("전처리 완료!")


# --- 4. 문제지(X)와 정답지(y) 재정의 ---
df['KOSPI_Target_Return'] = df['KOSPI'].pct_change().shift(-1)
df.dropna(inplace=True)
X = df.drop('KOSPI_Target_Return', axis=1)
y = df['KOSPI_Target_Return']
print(f"\n총 {len(X)}개의 데이터와 {len(X.columns)}개의 특성이 준비되었습니다.")
split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]


# --- 5. 하이퍼파라미터 튜닝 및 모델 훈련 ---
print("\n하이퍼파라미터 튜닝을 시작합니다... (시간 소요)")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("튜닝 및 훈련 완료!")
print(f"최적 파라미터: {grid_search.best_params_}")


# --- 6. 최종 예측 및 평가 ---
predicted_returns = best_model.predict(X_test)
predicted_prices = X_test['KOSPI'] * (1 + predicted_returns)
actual_prices = X_test['KOSPI'] * (1 + y_test)
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f"\n최종 모델 성능 평가 (Mean Absolute Error): {mae:.2f}")

# --- 7. 시각화 ---
if platform.system() == 'Windows': plt.rcParams['font.family'] = 'Malgun Gothic'
else: plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, actual_prices, label='Actual Price')
plt.plot(y_test.index, predicted_prices, label=f'Tuned Return-based Prediction (MAE: {mae:.2f})', linestyle='--')
plt.title('KOSPI Price Prediction by Forecasting Returns (Tuned)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# --- 8. 최종 모델 저장 ---
print("\n최종 모델을 'kospi_predictor.joblib' 파일로 저장합니다...")
joblib.dump(best_model, 'kospi_predictor.joblib')
print("모델 저장 완료!")