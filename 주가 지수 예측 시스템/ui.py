import streamlit as st
import requests
import pandas as pd

# --- 웹 화면 구성 ---
st.set_page_config(layout="wide") # 페이지 레이아웃을 넓게 설정
st.title("📈 KOSPI 지수 예측 시스템")
st.write("---")

# 화면을 두 개의 컬럼으로 분할
col1, col2 = st.columns([1, 2])

with col1:
    st.write("다양한 경제 지표를 바탕으로 다음 거래일의 KOSPI 종가를 예측합니다.")
    st.write("아래 버튼을 눌러 다음 거래일의 KOSPI 지수를 예측해보세요.")
    
    # 예측 버튼
    if st.button("🚀 다음 거래일 KOSPI 지수 예측하기"):
        # 버튼을 누르면 이전 결과 초기화
        if 'result' in st.session_state:
            del st.session_state['result']
        
        with st.spinner('최신 데이터를 수집하고 모델을 실행하는 중입니다... (약 30초 소요)'):
            try:
                # FastAPI 서버에 예측 요청 보내기
                response = requests.get("http://127.0.0.1:8000/predict")
                response.raise_for_status() # 200번대 상태 코드가 아니면 에러 발생
                
                # 세션 상태에 결과 저장
                st.session_state.result = response.json()

            except requests.exceptions.RequestException as e:
                st.error(f"API 서버에 연결할 수 없습니다. api.py 서버가 실행 중인지 확인해주세요.")
            except Exception as e:
                # FastAPI에서 보낸 상세 에러 메시지를 표시
                try:
                    detail = response.json().get('detail', str(e))
                    st.error(f"예측 중 에러가 발생했습니다: {detail}")
                except:
                    st.error(f"예측 중 에러가 발생했습니다: {e}")


with col2:
    # st.session_state에 결과가 있으면 화면에 표시
    if 'result' in st.session_state and st.session_state.result:
        res = st.session_state.result
        st.write(f"#### 🎯 예측 결과 ({res['prediction_date']})")
        
        # 결과를 3개의 컬럼으로 나눠서 표시
        res_col1, res_col2, res_col3 = st.columns(3)
        
        # [수정된 부분] 백엔드로부터 받은 모든 정보를 st.metric을 이용해 표시
        res_col1.metric(
            label="가장 최근 KOSPI 종가",
            value=f"{res['latest_actual_kospi_close']:,.2f} P"
        )
        
        res_col2.metric(
            label=f"예상 KOSPI 종가",
            value=f"{res['predicted_kospi_close']:,.2f} P",
            delta=f"{res['change_points']:,.2f} P ({res['change_percent']:.2f}%)"
        )
        
        # 세 번째 컬럼은 비워두거나 다른 정보를 추가할 수 있습니다.
        # res_col3.info("이 예측은 과거 데이터 기반이며, 실제 투자에 대한 조언이 아닙니다.")