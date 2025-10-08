from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import numpy as np
import traceback
import pandas_market_calendars as mcal # 휴장일 확인을 위해 추가

# --- 1. 설정 (Configuration) ---
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

# --- 2. FastAPI 앱 생성 및 모델 불러오기 ---
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load('kospi_predictor.joblib')
        print("모델을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print("저장된 모델 파일을 찾을 수 없습니다.")
    yield

app = FastAPI(title="KOSPI 예측 API", description="다양한 경제 지표를 바탕으로 다음 날의 KOSPI 종가를 예측하는 API입니다.", lifespan=lifespan)

# --- 3. API 엔드포인트 정의 ---
@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": model is not None}

@app.get("/predict")
async def predict():
    if model is None:
        raise HTTPException(status_code=500, detail="서버에 모델이 로드되지 않았습니다.")

    try:
        print("\n예측을 위한 최신 데이터 수집 및 전처리를 시작합니다...")
        
        END_DATE = pd.Timestamp.now()
        PREDICTION_START_DATE = END_DATE - pd.Timedelta(days=90)
        
        # 데이터 수집
        df_market = yf.download(list(YFINANCE_MAP.keys()), start=PREDICTION_START_DATE, end=END_DATE, progress=False, timeout=30)
        df_econ = web.DataReader(list(FRED_TICKERS.keys()), 'fred', PREDICTION_START_DATE, END_DATE)
        
        df_market = df_market['Close']
        df_market.rename(columns=YFINANCE_MAP, inplace=True)
        df_econ.columns = list(FRED_TICKERS.values())
        
        try:
            df_kor_1y = pd.read_excel('KOR_1Y_TREASURY.xlsx', index_col=0, parse_dates=True)
            df_kor_1y.columns = ['KOR_1Y_TREASURY']
        except FileNotFoundError:
            df_kor_1y = pd.DataFrame()

        # 데이터 통합 및 전처리
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
        print("데이터 준비 완료.")

        X_latest = df.tail(1)
        
        predicted_return = model.predict(X_latest)[0]
        
        latest_actual_kospi = X_latest['KOSPI'].iloc[0]
        predicted_price = latest_actual_kospi * (1 + predicted_return)
        
        print(f"예측된 KOSPI 종가: {predicted_price:.2f}")

        change_points = predicted_price - latest_actual_kospi
        change_percent = (change_points / latest_actual_kospi) * 100

        # --- [최종 수정된 부분] 다음 '개장일' 찾기 ---
        krx_calendar = mcal.get_calendar('XKRX')
        today = X_latest.index[0].date()
        # 오늘 다음 날부터 2주 후까지의 개장일 스케줄 조회
        future_schedule = krx_calendar.schedule(start_date=today + pd.Timedelta(days=1), end_date=today + pd.Timedelta(days=14))
        
        if not future_schedule.empty:
            next_trading_day = future_schedule.index[0]
        else:
            next_trading_day = today + pd.Timedelta(days=1) # 비상시 그냥 다음 날로 표시

        # 최종 결과 반환
        return {
            "prediction_date": next_trading_day.strftime('%Y-%m-%d'),
            "latest_actual_kospi_close": round(float(latest_actual_kospi), 2),
            "predicted_kospi_close": round(float(predicted_price), 2),
            "change_points": round(float(change_points), 2),
            "change_percent": round(float(change_percent), 2)
        }
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"예측 중 심각한 에러 발생: {error_details}")
        raise HTTPException(status_code=500, detail=f"예측을 처리하는 중 서버 에러가 발생했습니다: {str(e)}")