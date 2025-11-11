import streamlit as st
import requests
import pandas as pd
# --- [추가된 부분] Matplotlib 라이브러리 임포트 ---
import matplotlib.pyplot as plt
import platform
# --- [여기까지 추가] ---

# --- 한글 변환 맵 (이전과 동일) ---
FEATURE_NAME_MAP = {
    # 원본 (Targets)
    'KOSPI': 'KOSPI', 'KOSDAQ': 'KOSDAQ',
    'S&P500': 'S&P 500', 'NASDAQ': 'NASDAQ',
    # 원본 (Extras)
    'USD_KRW': '원/달러 환율', 'WTI_OIL': 'WTI 유가', 'GOLD': '금 가격',
    'DXY': '달러 인덱스', 'VIX': 'VIX 지수',
    # 원본 (FRED)
    'US_10Y_TREASURY': '미국 10년물 금리', 'US_3M_TREASURY': '미국 3개월물 금리',
    'KOR_10Y_TREASURY': '한국 10년물 금리', 'KOR_3M_TREASURY': '한국 3개월물 금리',
    'US_CPI': '미국 CPI', 'KOR_CPI': '한국 CPI',
    'US_Unemployment': '미국 실업률', 'KOR_Unemployment': '한국 실업률',
    'US_CSI': '미국 소비자동향지수', 'US_PPI': '미국 PPI',
    # 파생 (Yield Curve)
    'US_Yield_Curve': '미국 장단기 금리차', 'KOR_Yield_Curve': '한국 장단기 금리차',
}
derived_features = {}
original_features = list(FEATURE_NAME_MAP.keys())
for feature_name in original_features:
    korean_name = FEATURE_NAME_MAP.get(feature_name, feature_name)
    derived_features[f'{feature_name}_MA5'] = f'{korean_name} (5일 이동평균)'
    derived_features[f'{feature_name}_MA20'] = f'{korean_name} (20일 이동평균)'
    derived_features[f'{feature_name}_MA60'] = f'{korean_name} (60일 이동평균)'
    derived_features[f'{feature_name}_MA120'] = f'{korean_name} (120일 이동평균)'
    derived_features[f'{feature_name}_Momentum'] = f'{korean_name} (변동률)'
FEATURE_NAME_MAP.update(derived_features)
# --- [한글 변환 맵 끝] ---


# --- 웹 화면 구성 ---
st.set_page_config(layout="centered")
st.title("📈 주가 지수 예측 시스템")
st.write("---")

INDEX_OPTIONS = ['KOSPI', 'KOSDAQ', 'S&P500', 'NASDAQ']
selected_index = st.selectbox(
    "예측을 원하는 지수를 선택하세요:",
    INDEX_OPTIONS
)

if st.button(f"🚀 {selected_index} 다음 거래일 예측하기"):
    
    with st.spinner(f'{selected_index} 예측을 위해 데이터를 수집하고 모델을 실행하는 중입니다...'):
        try:
            # (이전 예측 API 호출 로직은 동일)
            predict_url = f"http://127.0.0.1:8000/predict/{selected_index.lower()}"
            response_predict = requests.get(predict_url, timeout=60)
            response_predict.raise_for_status()
            res = response_predict.json()
            st.session_state.predict_result = res
            
            features_url = f"http://127.0.0.1:8000/features/{selected_index.lower()}"
            response_features = requests.get(features_url, timeout=10)
            response_features.raise_for_status()
            features_data = response_features.json()
            st.session_state.features_data = features_data

        except requests.exceptions.RequestException as e:
            st.error(f"API 서버에 연결할 수 없습니다. api.py 서버가 실행 중인지 확인해주세요.")
            if 'predict_result' in st.session_state: del st.session_state['predict_result']
            if 'features_data' in st.session_state: del st.session_state['features_data']
        except Exception as e:
            try:
                detail = response_predict.json().get('detail', str(e))
                st.error(f"예측 중 에러가 발생했습니다: {detail}")
            except:
                 st.error(f"예측 중 에러가 발생했습니다: {e}")
            if 'predict_result' in st.session_state: del st.session_state['predict_result']
            if 'features_data' in st.session_state: del st.session_state['features_data']


# --- 예측 결과 표시 ---
if 'predict_result' in st.session_state and st.session_state.predict_result:
    # (이전 결과 표시 로직은 동일)
    res = st.session_state.predict_result
    st.write("---")
    st.write(f"#### 🎯 {res['index_name']} 예측 결과 ({res['prediction_date']})")
    col1, col2 = st.columns(2)
    col1.metric(
        label=f"가장 최근 {res['index_name']} 종가",
        value=f"{res['latest_actual_price']:,.2f} P"
    )
    col2.metric(
        label=f"예상 {res['index_name']} 종가",
        value=f"{res['predicted_price']:,.2f} P",
        delta=f"{res['change_points']:,.2f} P ({res['change_percent']:.2f}%)"
    )

# --- [수정된 부분] 예측 근거 표시 (Matplotlib 사용) ---
if 'features_data' in st.session_state and st.session_state.features_data:
    st.write("---")
    st.write(f"#### 📊 {selected_index} 예측에 사용된 Top 20 팩터 (by RandomForest)")
    
    features_data = st.session_state.features_data
    df_features = pd.DataFrame(features_data)
    
    # 1. Feature 컬럼의 이름을 한글로 변환
    df_features['Feature_Korean'] = df_features['Feature'].map(FEATURE_NAME_MAP).fillna(df_features['Feature'])
    
    # 2. 폰트 설정 (한글 깨짐 방지)
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin': # Mac OS
        plt.rcParams['font.family'] = 'AppleGothic'
    else: # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 방지

    # 3. Matplotlib으로 수평 막대 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 8)) # 차트 크기 (세로를 8로 길게)
    ax.barh(
        df_features['Feature_Korean'], 
        df_features['Importance']
    )
    ax.set_title(f"{selected_index} Top 20 팩터 중요도", fontsize=16)
    ax.set_xlabel("중요도 (Importance)")
    ax.invert_yaxis()  # 중요도가 높은 것이 위에 오도록
    plt.tight_layout() # 이름이 잘리지 않게 레이아웃 자동 조정

    # 4. Streamlit에 Matplotlib 차트 표시
    st.pyplot(fig)