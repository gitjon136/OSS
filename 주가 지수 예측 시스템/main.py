import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib 
import requests 
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- LSTM 모델 정의 ---
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# --- 1. 하이퍼파라미터 및 설정 ---
START_DATE_PD = '2020-01-01'
END_DATE_PD = pd.Timestamp.now().strftime('%Y-%m-%d')

# 하이퍼파라미터 사
sequence_length = 60
hidden_size = 128
num_layers = 2
output_size = 1
dropout_prob = 0.2
learning_rate = 0.001
num_epochs = 200
patience = 10
N_TOP_FEATURES = 30  # Bi-LSTM 훈련에 사용할 최정예 팩터 수
N_XAI_FEATURES = 20  # UI에 시각화할 팩터 수

# --- 2. 데이터 수집 (Configuration) ---
TARGET_TICKERS = {'^KS11': 'KOSPI', '^KQ11': 'KOSDAQ', '^GSPC': 'S&P500', '^IXIC': 'NASDAQ'}
EXTRA_TICKERS = {
    'KRW=X': 'USD_KRW', 'CL=F': 'WTI_OIL', 'GC=F': 'GOLD',
    'DX-Y.NYB': 'DXY', '^VIX': 'VIX'
}
FRED_TICKERS = {
    'DGS10': 'US_10Y_TREASURY', 'DGS3MO': 'US_3M_TREASURY',
    'IRLTLT01KRM156N': 'KOR_10Y_TREASURY', 'IR3TIB01KRM156N': 'KOR_3M_TREASURY',
    'CPIAUCSL': 'US_CPI', 'KORCPIALLMINMEI': 'KOR_CPI',
    'UNRATE': 'US_Unemployment', 'LRUNTTTTKRM156S': 'KOR_Unemployment',
    'UMCSENT': 'US_CSI', 'PPIACO': 'US_PPI'
}

# --- 3. 데이터 수집 및 전처리 ---
print("데이터를 수집하고 전처리합니다...")
all_market_tickers = list(TARGET_TICKERS.keys()) + list(EXTRA_TICKERS.keys())
df_market = yf.download(all_market_tickers, start=START_DATE_PD, end=END_DATE_PD, progress=False, timeout=15)['Close']
df_market.rename(columns={**TARGET_TICKERS, **EXTRA_TICKERS}, inplace=True)

df_econ = web.DataReader(list(FRED_TICKERS.keys()), 'fred', START_DATE_PD, END_DATE_PD)
df_econ.columns = list(FRED_TICKERS.values())

df = df_market.merge(df_econ, left_index=True, right_index=True, how='left')

df.index.name = 'DATE'
df['US_Yield_Curve'] = df['US_10Y_TREASURY'] - df['US_3M_TREASURY']
df['KOR_Yield_Curve'] = df['KOR_10Y_TREASURY'] - df['KOR_3M_TREASURY']
df.ffill(inplace=True); df.bfill(inplace=True)

for col in df.columns:
    df[f'{col}_MA5'] = df[col].rolling(window=5).mean()
    df[f'{col}_MA20'] = df[col].rolling(window=20).mean()
    df[f'{col}_MA60'] = df[col].rolling(window=60).mean()
    df[f'{col}_MA120'] = df[col].rolling(window=120).mean()
    df[f'{col}_Momentum'] = df[col].pct_change()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True); df.bfill(inplace=True)
print("전처리 완료!")

# --- 4. 각 지수별로 모델 훈련 및 저장 ---
for ticker_code, ticker_name in TARGET_TICKERS.items():
    print(f"\n--- {ticker_name} 모델 훈련 시작 ---")
    
    # 모델별 데이터프레임 및 특성 선택
    all_features = df.copy()
    if ticker_name in ['S&P500', 'NASDAQ']:
        korea_specific_cols = [col for col in all_features.columns if 'KOR' in col or 'KOSPI' in col or 'KOSDAQ' in col]
        features_to_use = [col for col in all_features.columns if col not in korea_specific_cols]
        temp_df = all_features[features_to_use]
        print(f"{ticker_name} 모델은 한국 관련 특성 {len(korea_specific_cols)}개를 제외하고 학습합니다.")
    else:
        temp_df = all_features
    
    temp_df[f'Target_Return'] = temp_df[ticker_name].pct_change().shift(-1)
    temp_df.dropna(inplace=True)
    X_2d_all = temp_df.drop(f'Target_Return', axis=1) # 모든 팩터
    y_2d = temp_df[f'Target_Return']

    # --- 4-1. 랜덤 포레스트로 '예측 근거' 및 '최정예 팩터' 추출 ---
    print(f"[{ticker_name}] RandomForest 분석기로 특성 중요도 분석 중...")
    split_point_rf = int(len(X_2d_all) * 0.8) 
    X_train_rf, y_train_rf = X_2d_all.iloc[:split_point_rf], y_2d.iloc[:split_point_rf]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)
    
    importances = rf_model.feature_importances_
    feature_names = X_2d_all.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # '예측 근거' CSV 저장 (Top 20)
    top_xai_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(N_XAI_FEATURES)
    csv_filename = f'{ticker_name.lower()}_features.csv'
    top_xai_features.to_csv(csv_filename, index=False)
    print(f"'{csv_filename}' 파일로 특성 중요도(XAI) 저장 완료!")

    # '최정예 팩터' 리스트 추출 (Top 30)
    top_model_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(N_TOP_FEATURES)['Feature'].tolist()
    if ticker_name not in top_model_features:
        top_model_features.append(ticker_name) 
    print(f"[{ticker_name}] Bi-LSTM 훈련에 사용할 최정예 팩터 {len(top_model_features)}개를 선별했습니다.")
    
    # --- 4-2. '최정예 팩터'로 딥러닝 데이터 전처리 ---
    X_2d_selected = X_2d_all[top_model_features] # Top 30개로만 X데이터 재구성
    
    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_2d_selected); y_scaled = scaler_y.fit_transform(y_2d.values.reshape(-1, 1))
    joblib.dump(scaler_X, f'scaler_X_{ticker_name.lower()}.joblib')
    joblib.dump(scaler_y, f'scaler_y_{ticker_name.lower()}.joblib')
    
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_scaled[i+sequence_length])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    X_tensor = torch.FloatTensor(X_seq); y_tensor = torch.FloatTensor(y_seq)
    train_size = int(len(X_tensor) * 0.7); val_size = int(len(X_tensor) * 0.15)
    X_train, X_val, X_test = X_tensor[:train_size], X_tensor[train_size:train_size+val_size], X_tensor[train_size+val_size:]
    y_train, y_val, y_test = y_tensor[:train_size], y_tensor[train_size:train_size+val_size], y_tensor[train_size+val_size:]
    input_size = X_train.shape[2] # input_size는 30(N_TOP_FEATURES)이 됨

    # --- 파라미터로 최종 모델 훈련 ---
    model = BiLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model_path = f'{ticker_name.lower()}_predictor.pth'
    print(f"[{ticker_name}] 최종 훈련을 시작합니다...")
    
    best_val_loss = float('inf'); epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train(); outputs = model(X_train); optimizer.zero_grad()
        loss = criterion(outputs, y_train); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val); val_loss = criterion(val_outputs, y_val)
        if (epoch+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss; epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"성능 개선이 없어 {epoch+1}번째 Epoch에서 훈련을 조기 종료합니다.")
            break
    print(f"{ticker_name} 모델 훈련 완료!")

    # --- 4-4. 최종 성능 평가 (백테스팅) ---
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test)
    predicted_returns = scaler_y.inverse_transform(predictions_scaled.numpy()).flatten()
    y_test_original_returns = scaler_y.inverse_transform(y_test.numpy()).flatten()
    
    test_start_index = train_size + val_size + sequence_length
    X_test_original_selected = X_2d_selected.iloc[test_start_index:]
    
    actual_prices = X_test_original_selected[ticker_name] * (1 + y_test_original_returns)
    predicted_prices = X_test_original_selected[ticker_name] * (1 + predicted_returns)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    print(f"-> {ticker_name} 모델의 최종 MAE: {mae:.2f}")
    print(f"'{model_path}' 파일로 모델 저장 완료!")

    # --- 4-5. 백테스팅 결과 저장 (UI용) ---
    print(f"--- {ticker_name} 모델의 백테스팅 결과 저장 중 ---")
    df_backtest = pd.DataFrame({
        'Date': actual_prices.index,
        'Actual_Price': actual_prices,
        'Predicted_Price': predicted_prices
    })
    backtest_filename = f'{ticker_name.lower()}_backtest.csv'
    df_backtest.to_csv(backtest_filename, index=False)
    print(f"'{backtest_filename}' 파일로 백테스팅 결과 저장 완료!")

print("\n--- 모든 딥러닝 모델 훈련 및 저장이 완료되었습니다. ---")