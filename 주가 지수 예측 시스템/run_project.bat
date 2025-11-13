@echo off
REM -----------------------------------------------------------------
REM           XAI 기반 주가 지수 예측 시스템 - 원클릭 실행 스크립트
REM           (v3.0 - 절대 경로 지정)
REM -----------------------------------------------------------------
title XAI Prediction System Launcher

echo [알림] 2개의 서버를 시작합니다...
echo.

REM --- 1. 백엔드 (FastAPI) 서버 시작 ---
REM "python" 대신, 우리가 라이브러리를 설치한 "python3.10.exe"의 '절대 경로'를 사용합니다.
echo [1/2] 백엔드 (FastAPI) 서버를 새 창에서 시작합니다... (http://127.0.0.1:8000)
set PYTHON_EXE="C:\Users\Yeongjin\AppData\Local\Microsoft\WindowsApps\python3.10.exe"
start "XAI_Backend_API" %PYTHON_EXE% -m uvicorn api:app

echo    ... 백엔드 서버가 켜지는 중입니다. (3초 대기)
timeout /t 3
echo.

REM --- 2. 프론트엔드 (Streamlit) 서버 시작 ---
echo [2/2] 프론트엔드 (Streamlit) UI를 새 창에서 시작합니다... (http://localhost:8501)
start "XAI_Frontend_UI" %PYTHON_EXE% -m streamlit run ui.py

echo.
echo -----------------------------------------------------------------
echo [성공] 2개의 서버가 시작되었습니다.
echo 잠시 후, Streamlit UI가 웹 브라우저에서 자동으로 열립니다.
echo (실행된 2개의 검은색 터미널 창을 닫으면 서버가 종료됩니다.)
echo -----------------------------------------------------------------
pause