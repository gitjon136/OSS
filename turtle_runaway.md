# Turtle Runaway - 설명서

## 1. 개요
플레이어(Chaser)를 조작하여 스마트하게 움직이는 거북이(Runner)를 잡는 게임입니다.

---

## 2. 요구 사항 충족

| 요구 사항 | 구현 내용 |
|-----------|-----------|
| Timer | 상단에 남은 시간을 표시 (`Time: xx / Score: xx`) |
| Intelligent Turtle | Runner가 Chaser 입력을 분석하고 반대 방향으로 도망가는 AI 구현 |
| Scoring System | Chaser가 Runner를 잡으면 점수 증가 |
| Optional | 창 제목 “Turtle Runaway” |

---

## 3. 주요 구현 내용

### 3.1 Chaser 조작
- 방향키로 이동 가능
- 이동 시 heading도 변경
- 화면 경계 내에서만 이동

### 3.2 Runner AI
- 최근 Chaser 키 입력 5개 기억
- 입력 방향과 반대 방향으로 도망
- 입력 없으면 랜덤 회피
- 화면 경계 내 이동

### 3.3 점수 및 캐치 처리
- 일정 거리 내 접근 시 점수 증가
- 캐치 후 Runner 위치 랜덤 재배치 + AI 이동 1프레임 정지

### 3.4 Timer
- 게임 시작 시 60초 설정
- 매 step마다 남은 시간 표시, 0되면 종료

### 3.5 화면 경계 처리
- Chaser, Runner 모두 화면 밖 이동 불가
- X/Y 좌표 범위: -350 ~ 350

---

## 4. 실행 방법
1. `turtle_runaway.py` 실행
2. 방향키로 Red 거북이(Chaser) 조작
3. Blue 거북이(Runner)가 반대 방향으로 도망
4. Runner 잡으면 점수 증가, 제한 시간 종료 시 최종 점수 표시

---

## 5. 결과 화면
- 게임 실행 시 상단에 **남은 시간 / 점수** 표시
- Runner 캐치 후 위치 재배치
- 게임 종료 시 시간 종료를 알리고 점수 표시