import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import platform

# --- [한글 변환 맵] ---
FEATURE_NAME_MAP = {
    'KOSPI': 'KOSPI (오늘 종가)', 'KOSDAQ': 'KOSDAQ (오늘 종가)',
    'S&P500': 'S&P 500 (오늘 종가)', 'NASDAQ': 'NASDAQ (오늘 종가)',
    'USD_KRW': '원/달러 환율', 'WTI_OIL': 'WTI 유가', 'GOLD': '금 가격',
    'DXY': '달러 인덱스(DXY)', 'VIX': '변동성 지수(VIX)',
    'US_10Y_TREASURY': '미국 10년물 금리', 'US_3M_TREASURY': '미국 3개월물 금리',
    'KOR_10Y_TREASURY': '한국 10년물 금리', 'KOR_3M_TREASURY': '한국 3개월물 금리',
    'US_CPI': '미국 CPI', 'KOR_CPI': '한국 CPI',
    'US_Unemployment': '미국 실업률', 'KOR_Unemployment': '한국 실업률',
    'US_CSI': '미국 소비자동향지수', 'US_PPI': '미국 PPI',
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
# --- [맵 끝] ---

# --- 웹 화면 구성 ---
st.set_page_config(layout="wide")
st.title("📈 다중 팩터 기반 주가 지수 예측 시스템")
st.write("---")

INDEX_OPTIONS = ['KOSPI', 'KOSDAQ', 'S&P500', 'NASDAQ']
selected_index = st.selectbox("예측을 원하는 지수를 선택하세요:", INDEX_OPTIONS)

if st.button(f"🚀 {selected_index} 다음 거래일 예측하기"):
    
    with st.spinner(f'{selected_index} 예측을 위해 모든 데이터를 수집하고 분석하는 중입니다...'):
        try:
            base_url = "http://127.0.0.1:8000"
            
            # --- 4개의 API를 동시에 호출 ---
            response_predict = requests.get(f"{base_url}/predict/{selected_index.lower()}", timeout=60)
            response_features = requests.get(f"{base_url}/features/{selected_index.lower()}", timeout=10)
            response_chart = requests.get(f"{base_url}/chart/{selected_index.lower()}", timeout=10)
            response_backtest = requests.get(f"{base_url}/backtest/{selected_index.lower()}", timeout=10)
            
            response_predict.raise_for_status()
            response_features.raise_for_status()
            response_chart.raise_for_status()
            response_backtest.raise_for_status()
            
            st.session_state.predict_result = response_predict.json()
            st.session_state.features_data = response_features.json()
            st.session_state.chart_data = response_chart.json()
            st.session_state.backtest_data = response_backtest.json()

        except requests.exceptions.RequestException as e:
            st.error(f"API 서버에 연결할 수 없습니다. api.py 서버가 실행 중인지 확인해주세요.")
            if 'predict_result' in st.session_state: del st.session_state['predict_result']
        except Exception as e:
            st.error(f"예측 중 에러가 발생했습니다: {e}")
            if 'predict_result' in st.session_state: del st.session_state['predict_result']


# --- 탭(Tab)을 이용한 결과 표시 ---
if 'predict_result' in st.session_state and st.session_state.predict_result:
    res = st.session_state.predict_result
    
    st.write("---")
    st.write(f"#### 🎯 {res['index_name']} 예측 결과 ({res['prediction_date']})")
    
    col1, col2 = st.columns(2)
    col1.metric(label=f"가장 최근 {res['index_name']} 종가", value=f"{res['latest_actual_price']:,.2f} P")
    col2.metric(label=f"예상 {res['index_name']} 종가", value=f"{res['predicted_price']:,.2f} P",
                delta=f"{res['change_points']:,.2f} P ({res['change_percent']:.2f}%)")

    tab1, tab2, tab3 = st.tabs(["📊 일봉 차트 (최근 6개월)", "🧠 예측 근거 (Top 20 팩터)", "📈 과거 예측 성과 (Backtest)"])

    with tab1:
        if 'chart_data' in st.session_state:
            df_chart = pd.DataFrame(st.session_state.chart_data)
            df_chart['Date'] = pd.to_datetime(df_chart['Date'])
            
            fig = go.Figure(data=[go.Candlestick(
                x=df_chart['Date'], open=df_chart['Open'], high=df_chart['High'],
                low=df_chart['Low'], close=df_chart['Close'],
                increasing_line_color='red', decreasing_line_color='blue',
                name=res['index_name']
            )])
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA5'], mode='lines', name='5일 이동평균', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA20'], mode='lines', name='20일 이동평균', line=dict(color='green', width=1)))
            fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MA60'], mode='lines', name='60일 이동평균', line=dict(color='purple', width=1)))
            
            fig.update_layout(title=f"{res['index_name']} 일봉 차트 (6개월) + 이동평균선",
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if 'features_data' in st.session_state:
            st.write(f"#### {selected_index} 예측에 사용된 Top 20 팩터 (by RandomForest)")
            features_data = st.session_state.features_data
            df_features = pd.DataFrame(features_data)
            df_features['Feature_Korean'] = df_features['Feature'].map(FEATURE_NAME_MAP).fillna(df_features['Feature'])
            
            if platform.system() == 'Windows': plt.rcParams['font.family'] = 'Malgun Gothic'
            else: plt.rcParams['font.family'] = 'NanumGothic'
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(df_features['Feature_Korean'], df_features['Importance'])
            ax.set_title(f"{selected_index} Top 20 팩터 중요도", fontsize=16)
            ax.set_xlabel("중요도 (Importance)")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            
    with tab3:
        if 'backtest_data' in st.session_state:
            st.write(f"#### {selected_index} 모델 과거 예측 성과 (테스트 기간)")
            df_backtest = pd.DataFrame(st.session_state.backtest_data)
            df_backtest['Date'] = pd.to_datetime(df_backtest['Date'])
            
            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(x=df_backtest['Date'], y=df_backtest['Actual_Price'],
                                            mode='lines', name='실제 가격 (Actual)',
                                            line=dict(color='blue'))) 
            
            fig_backtest.add_trace(go.Scatter(x=df_backtest['Date'], y=df_backtest['Predicted_Price'],
                                            mode='lines', name='모델 예측 가격 (Predicted)',
                                            line=dict(color='red', dash='dash'))) 
            
            fig_backtest.update_layout(title=f"{res['index_name']} 예측 정확도 백테스팅",
                                       xaxis_title="날짜", yaxis_title="지수")
            st.plotly_chart(fig_backtest, use_container_width=True)