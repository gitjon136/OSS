import pandas as pd
import matplotlib.pyplot as plt

# CSV 불러오기 (파일명은 실제 파일명으로 변경)
df = pd.read_csv("data.csv")

# 시각화에 필요한 열만 선택
data = df[["전혀 동의하지 않는다 (%)", "대체로 동의하지 않는다 (%)", 
           "대체로 동의 한다 (%)", "전적으로 동의 한다 (%)"]]

# 질문 항목 (행 이름)
questions = df.iloc[:, 0]

# 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 6))

# 100% 누적 막대그래프
data.plot(kind="barh",
          stacked=True,
          ax=ax,
          color=["#4B8BBE", "#306998", "#FFD43B", "#FF6F61"])

# y축 라벨을 질문 항목으로 교체
ax.set_yticks(range(len(questions)))
ax.set_yticklabels(questions)

# 범례
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)

# 제목
ax.set_title("미디어 관련 인식 조사 결과 (100% 누적)", fontsize=14, fontweight="bold")

# x축 퍼센트 표시
ax.set_xlabel("응답 비율 (%)")

plt.tight_layout()
plt.show()
