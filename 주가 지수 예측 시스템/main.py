import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- 1. 설정 (Configuration) ---
START_DATE = '2020-01-01'
KOSPI_TICKER = '^KS11'
SNP500_TICKER = '^GSPC'

# --- 2. 데이터 수집 (Data Collection) ---
print("데이터 수집을 시작합니다...")
# yfinance는 여러 티커를 동시에 요청할 수 있습니다.
all_df = yf.download([KOSPI_TICKER, SNP500_TICKER], start=START_DATE, progress=False, timeout=15)

if all_df.empty:
    print("[에러] 데이터 다운로드에 실패했습니다.")
    print("네트워크 환경을 변경(예: 핫스팟 사용)한 후 다시 시도해주세요.")
    sys.exit()

print("데이터 수집 완료!")


# --- 3. 데이터 전처리 및 특성 공학 ---
print("\n데이터 전처리 및 특성 공학을 시작합니다...")

# [기존] 'Close' 데이터만 추출 및 컬럼 이름 변경
df = all_df['Close']
df.columns = ['KOSPI_Close', 'SNP500_Close']

# --- [추가된 부분] 특성 공학 ---
# 1. 이동 평균(Moving Average) 추가: 5일, 20일 이동평균선
df['KOSPI_MA5'] = df['KOSPI_Close'].rolling(window=5).mean()
df['KOSPI_MA20'] = df['KOSPI_Close'].rolling(window=20).mean()
df['SNP500_MA5'] = df['SNP500_Close'].rolling(window=5).mean()
df['SNP500_MA20'] = df['SNP500_Close'].rolling(window=20).mean()

# 2. 전일 대비 변동률(Momentum) 추가
df['KOSPI_Momentum'] = df['KOSPI_Close'].pct_change()
df['SNP500_Momentum'] = df['SNP500_Close'].pct_change()

# 결측치 처리 (fillna)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
print("특성 공학 및 결측치 처리 완료!")

# --- 문제지(X)와 정답지(y) 분리 ---

# 정답(y) 데이터 정의: 다음 날의 KOSPI 종가
df['KOSPI_Target'] = df['KOSPI_Close'].shift(-1)

# 맨 마지막 날의 Target은 비어있으므로, 해당 행을 제거
df.dropna(inplace=True)

# 문제지(X)와 정답지(y)를 최종적으로 분리
X = df.drop('KOSPI_Target', axis=1)
y = df['KOSPI_Target']

print("\n--- 문제지 (X 데이터 샘플) ---")
print(X.tail())
print("\n--- 정답지 (y 데이터 샘플) ---")
print(y.tail())


# --- 4. 데이터 시각화 (Subplot) ---
print("\n데이터 시각화를 시작합니다...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
fig.suptitle('KOSPI vs S&P 500 Index (Subplots)', fontsize=16)

axes[0].plot(df.index, df['KOSPI_Close'], color='blue', label='KOSPI Close')
axes[0].set_ylabel('KOSPI Index')
axes[0].legend(loc='upper left')
axes[0].grid(True)

axes[1].plot(df.index, df['SNP500_Close'], color='green', label='S&P 500 Close')
axes[1].set_ylabel('S&P 500 Index')
axes[1].set_xlabel('Date')
axes[1].legend(loc='upper left')
axes[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

print("\n모든 작업이 완료되었습니다!")