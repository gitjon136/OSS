from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware # CORS를 위해 추가
import joblib
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import numpy as np
import traceback
import pandas_market_calendars as mcal
import torch
import torch.nn as nn
import requests 
import warnings
import os

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 1. LSTM 모델 정의 (main.py와 동일하게) ---
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# --- 2. 하이퍼파라미터 (main.py와 동일하게) ---
sequence_length = 60
hidden_size = 128
num_layers = 2
output_size = 1
dropout_prob = 0.2

# --- 3. 데이터 수집용 설정 ---
TARGET_TICKERS = {'KOSPI': '^KS11', 'KOSDAQ': '^KQ11', 'S&P500': '^GSPC', 'NASDAQ': '^IXIC'}
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

# --- 4. FastAPI 앱 및 모델 로딩 ---
models = {}
scalers_X = {}
scalers_y = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # (이전과 동일한 모델 로딩 코드)
    # ...
    yield

app = FastAPI(title="주가 지수 예측 API (Bi-LSTM ver.)", lifespan=lifespan)

# --- [추가된 부분] CORS 미들웨어 설정 ---
# Streamlit 기본 주소(localhost:8501)에서의 요청을 허용합니다.
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # 모든 HTTP 메소드 허용
    allow_headers=["*"], # 모든 HTTP 헤더 허용
)

# --- 5. API 엔드포인트 ---
@app.get("/")
def read_root():
    return {"status": "online", "models_loaded": list(models.keys())}

@app.get("/predict/{index_name}")
async def predict(index_name: str):
    # (이하 /predict 엔드포인트 코드는 이전과 동일합니다)
    # ...
    index_name_upper = index_name.upper()
    if index_name_upper not in models:
        raise HTTPException(status_code=404, detail="모델이 로드되지 않았습니다.")
    
    model = models[index_name_upper]
    scaler_X = scalers_X[index_name_upper]
    scaler_y = scalers_y[index_name_upper]

    try:
        print(f"\n[{index_name_upper}] 예측을 위한 최신 데이터 수집 및 전처리를 시작합니다...")
        
        END_DATE_PD = pd.Timestamp.now()
        START_DATE_PD = END_DATE_PD - pd.Timedelta(days=180 + sequence_length)
        
        all_market_tickers = list(TARGET_TICKERS.values()) + list(EXTRA_TICKERS.keys())
        df_market = yf.download(all_market_tickers, start=START_DATE_PD, end=END_DATE_PD, progress=False, timeout=30)['Close']
        df_market.rename(columns={v: k for k, v in TARGET_TICKERS.items()}, inplace=True)
        df_market.rename(columns=EXTRA_TICKERS, inplace=True)

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
        print("데이터 준비 완료.")
        
        features_to_use = list(scaler_X.feature_names_in_)
        if index_name_upper in ['S&P500', 'NASDAQ']:
            korea_specific_cols = [col for col in df.columns if 'KOR' in col or 'KOSPI' in col or 'KOSDAQ' in col]
            final_features = [feat for feat in features_to_use if feat not in korea_specific_cols]
            df_final = df[final_features]
        else:
            df_final = df[features_to_use]
        
        X_latest_df = df_final.tail(sequence_length)
        X_latest_scaled = scaler_X.transform(X_latest_df)
        X_latest_tensor = torch.FloatTensor(X_latest_scaled).unsqueeze(0)
        
        print("딥러닝 모델 예측을 수행합니다...")
        model.eval()
        with torch.no_grad():
            prediction_scaled = model(X_latest_tensor)
        
        predicted_return = scaler_y.inverse_transform(prediction_scaled.numpy())[0][0]
        
        latest_actual_price = X_latest_df[index_name_upper].iloc[-1]
        predicted_price = latest_actual_price * (1 + predicted_return)
        
        change_points = predicted_price - latest_actual_price
        change_percent = (change_points / latest_actual_price) * 100
        
        ticker_code = TARGET_TICKERS[index_name_upper]
        calendar_name = 'XKRX' if index_name_upper in ['KOSPI', 'KOSDAQ'] else 'NYSE'
        calendar = mcal.get_calendar(calendar_name)
        today = X_latest_df.index[-1].date()
        future_schedule = calendar.schedule(start_date=today + pd.Timedelta(days=1), end_date=today + pd.Timedelta(days=14))
        next_trading_day = future_schedule.index[0] if not future_schedule.empty else today + pd.Timedelta(days=1)

        print(f"[{index_name_upper}] 예측 완료: {predicted_price:.2f}")

        return {
            "index_name": index_name_upper,
            "prediction_date": next_trading_day.strftime('%Y-%m-%d'),
            "latest_actual_price": round(float(latest_actual_price), 2),
            "predicted_price": round(float(predicted_price), 2),
            "change_points": round(float(change_points), 2),
            "change_percent": round(float(change_percent), 2)
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"예측 중 심각한 에러 발생: {error_details}")
        raise HTTPException(status_code=500, detail=f"예측을 처리하는 중 서버 에러가 발생했습니다: {str(e)}")