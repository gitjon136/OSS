from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
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

# --- 3. 데이터 수집용 설정 (main.py와 동일하게) ---
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
    global models, scalers_X, scalers_y
    print("서버 시작... 딥러닝 모델과 스케일러를 불러옵니다.")
    for name in TARGET_TICKERS.keys():
        try:
            scaler_X_path = f'scaler_X_{name.lower()}.joblib'
            scaler_y_path = f'scaler_y_{name.lower()}.joblib'
            model_path = f'{name.lower()}_predictor.pth'
            
            scalers_X[name] = joblib.load(scaler_X_path)
            scalers_y[name] = joblib.load(scaler_y_path)
            
            input_size = scalers_X[name].n_features_in_
            model_instance = BiLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
            
            model_instance.load_state_dict(torch.load(model_path))
            model_instance.eval()
            models[name] = model_instance
            print(f"[{name}] 모델 및 스케일러 로딩 성공.")
        except FileNotFoundError as e:
            print(f"[치명적 에러] {name} 모델 로딩 실패: {e.filename}")
    yield

app = FastAPI(title="주가 지수 예측 API (Bi-LSTM ver.)", lifespan=lifespan)

# --- 5. API 엔드포인트 (/predict) ---
@app.get("/")
def read_root():
    return {"status": "online", "models_loaded": list(models.keys())}

@app.get("/predict/{index_name}")
async def predict(index_name: str):
    index_name_upper = index_name.upper()
    if index_name_upper not in models:
        raise HTTPException(status_code=404, detail="모델이 로드되지 않았습니다.")
    
    # (이하 예측 로직은 이전과 동일)
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
        df.fillna(0, inplace=True)
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
        
        if np.isnan(predicted_return):
            raise HTTPException(status_code=500, detail="모델이 'nan'을 예측했습니다.")
            
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

# --- 6. [추가된 부분] API 엔드포인트 (/features) ---
@app.get("/features/{index_name}")
async def get_features(index_name: str):
    index_name_upper = index_name.upper()
    if index_name_upper not in TARGET_TICKERS:
        raise HTTPException(status_code=404, detail="지원하지 않는 지수입니다.")
    
    csv_filename = f'{index_name_upper.lower()}_features.csv'
    
    try:
        # 1. main.py가 저장한 CSV 파일을 읽음
        df_features = pd.read_csv(csv_filename)
        
        # 2. DataFrame을 JSON (dict list)으로 변환하여 반환
        return df_features.to_dict('records')
    
    except FileNotFoundError:
        print(f"[에러] {csv_filename} 파일을 찾을 수 없습니다.")
        raise HTTPException(status_code=404, detail=f"'{csv_filename}' 파일을 찾을 수 없습니다. main.py를 실행하여 생성해주세요.")
    except Exception as e:
        print(f"특성 파일 처리 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail="특성 파일을 처리하는 중 에러가 발생했습니다.")