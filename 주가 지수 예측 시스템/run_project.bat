@echo off
REM -----------------------------------------------------------------
REM           XAI 기반 주가 지수 예측 시스템 - 원클릭 실행 스크립트
REM           (v4.0 - 상대 경로)
REM -----------------------------------------------------------------
title Prediction System Launcher

echo [알림] 시스템을 시작합니다...
echo [알림] 실행 전 'pip install -r requirements.txt'로 라이브러리를 설치해야 합니다.
echo.

REM --- 1. 이 .bat 파일이 있는 폴더(프로젝트 폴더)로 강제 이동 ---
cd /d "%~dp0"
echo [INFO] 현재 작업 폴더: %cd%
echo.

REM --- 2. 백엔드 (FastAPI) 서버 시작 ---
echo [1/2] 백엔드 (FastAPI) 서버를 새 창에서 시작합니다... (http://127.0.0.1:8000)
start "Backend_API" python -m uvicorn api:app

echo    ... 백엔드 서버가 켜지는 중입니다. (3초 대기)
timeout /t 3

REM --- 3. 프론트엔드 (Streamlit) 서버 시작 ---
echo [2/2] 프론트엔드 (Streamlit) UI를 새 창에서 시작합니다... (http://localhost:8501)
start "Frontend_UI" python -m streamlit run ui.py

echo.
echo [성공] 2개의 서버가 시작되었습니다.
echo 잠시 후, Streamlit UI가 웹 브라우저에서 자동으로 열립니다.
echo (실행된 2개의 검은색 터미널 창을 닫으면 서버가 종료됩니다.)
echo -----------------------------------------------------------------
pause