import streamlit as st
import requests
import pandas as pd

# --- 웹 화면 구성 ---
st.set_page_config(layout="centered") # 페이지 레이아웃을 중앙 정렬로 설정
st.title("📈 주가 지수 예측 시스템")
st.write("---")

# 예측할 지수 목록
INDEX_OPTIONS = ['KOSPI', 'KOSDAQ', 'S&P500', 'NASDAQ']

# --- [수정된 부분] ---
# 1. 드롭다운 메뉴로 예측할 지수 1개 선택
selected_index = st.selectbox(
    "예측을 원하는 지수를 선택하세요:",
    INDEX_OPTIONS
)

st.write("아래 버튼을 누르면 선택한 지수의 다음 거래일을 예측합니다.")

# 2. 선택된 지수만 예측하는 버튼
if st.button(f"🚀 {selected_index} 다음 거래일 예측하기"):
    
    with st.spinner(f'{selected_index} 예측을 위해 데이터를 수집하고 모델을 실행하는 중입니다...'):
        try:
            # 3. 선택된 지수에 맞춰 API 요청
            api_url = f"http://127.0.0.1:8000/predict/{selected_index.lower()}"
            response = requests.get(api_url, timeout=60) # 타임아웃 60초
            response.raise_for_status()
            
            res = response.json()
            
            # 4. 예측 결과 표시 (컬럼 사용)
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

        except requests.exceptions.RequestException:
            st.error(f"API 서버에 연결할 수 없습니다. api.py 서버가 실행 중인지 확인해주세요.")
        except Exception as e:
            try:
                detail = response.json().get('detail', '알 수 없는 에러')
                st.error(f"예측 중 에러가 발생했습니다: {detail}")
            except:
                 st.error(f"예측 중 에러가 발생했습니다: {e}")