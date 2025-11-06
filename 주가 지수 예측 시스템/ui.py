import streamlit as st
import requests
import pandas as pd

# --- 웹 화면 구성 ---
st.set_page_config(layout="centered") # 페이지 레이아웃을 중앙 정렬로 설정
st.title("📈 주가 지수 예측 시스템")
st.write("---")

# 예측할 지수 목록
INDEX_OPTIONS = ['KOSPI', 'KOSDAQ', 'S&P500', 'NASDAQ']

st.write("아래 버튼을 누르면 4대 주요 지수의 다음 거래일을 한 번에 예측합니다.")

# 예측 버튼
if st.button("🚀 모든 지수 예측하기"):
    # 이전 결과 초기화
    if 'results' in st.session_state:
        del st.session_state['results']
    
    results = {}
    progress_bar = st.progress(0, text="예측을 시작합니다...")

    # 각 지수별로 API 요청을 보내고 결과 저장
    for i, index_name in enumerate(INDEX_OPTIONS):
        progress_text = f"{index_name} 예측을 위해 데이터를 수집하고 모델을 실행하는 중입니다..."
        progress_bar.progress((i + 0.5) / len(INDEX_OPTIONS), text=progress_text)
        
        try:
            api_url = f"http://127.0.0.1:8000/predict/{index_name.lower()}"
            response = requests.get(api_url, timeout=60) # 타임아웃을 60초로 넉넉하게 설정
            response.raise_for_status()
            results[index_name] = response.json()
        except requests.exceptions.RequestException:
            st.error(f"{index_name} 예측 실패: API 서버에 연결할 수 없습니다. api.py 서버가 실행 중인지 확인해주세요.")
            results = {} # 하나라도 실패하면 중단
            break
        except Exception:
            try:
                detail = response.json().get('detail', '알 수 없는 에러')
                st.error(f"{index_name} 예측 실패: {detail}")
            except:
                 st.error(f"{index_name} 예측 실패: 알 수 없는 에러가 발생했습니다.")
            results = {} # 하나라도 실패하면 중단
            break
    
    if results:
        progress_bar.progress(1.0, text="모든 예측이 완료되었습니다!")
        st.session_state.results = results


# st.session_state에 결과가 있으면 화면에 표시
if 'results' in st.session_state and st.session_state.results:
    st.write("---")
    st.write("#### 🎯 예측 결과")

    # --- [수정된 부분] 2x2 그리드 생성 ---
    # 1. 첫 번째 행 (KOSPI, KOSDAQ)
    top_col1, top_col2 = st.columns(2)
    
    with top_col1:
        res = st.session_state.results.get('KOSPI')
        if res:
            st.metric(
                label=f"KOSPI ({res['prediction_date']})",
                value=f"{res['predicted_price']:,.2f} P",
                delta=f"{res['change_points']:,.2f} P ({res['change_percent']:.2f}%)"
            )

    with top_col2:
        res = st.session_state.results.get('KOSDAQ')
        if res:
            st.metric(
                label=f"KOSDAQ ({res['prediction_date']})",
                value=f"{res['predicted_price']:,.2f} P",
                delta=f"{res['change_points']:,.2f} P ({res['change_percent']:.2f}%)"
            )

    # 2. 두 번째 행 (S&P 500, NASDAQ)
    bottom_col1, bottom_col2 = st.columns(2)

    with bottom_col1:
        res = st.session_state.results.get('S&P500')
        if res:
            st.metric(
                label=f"S&P 500 ({res['prediction_date']})",
                value=f"{res['predicted_price']:,.2f} P",
                delta=f"{res['change_points']:,.2f} P ({res['change_percent']:.2f}%)"
            )

    with bottom_col2:
        res = st.session_state.results.get('NASDAQ')
        if res:
            st.metric(
                label=f"NASDAQ ({res['prediction_date']})",
                value=f"{res['predicted_price']:,.2f} P",
                delta=f"{res['change_points']:,.2f} P ({res['change_percent']:.2f}%)"
            )