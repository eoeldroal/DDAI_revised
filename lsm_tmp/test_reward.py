# 파일명: test_reward.py
import sys
import os

# 현재 경로가 verl 루트라고 가정 (필요시 경로 수정)
sys.path.append(os.getcwd())

try:
    from verl.utils.reward_score import default_compute_score
    print(">>> 1. 함수 import 성공")
except ImportError as e:
    print(f">>> [치명적 오류] import 실패: {e}")
    sys.exit(1)

# 테스트 데이터
data_source = "slidevqa_train_6667"  # 문제가 되었던 그 이름
solution_str = "This is a test answer."
ground_truth = "This is a test answer."
extra_info = {"image": "dummy_image"} # vrag 내부에서 extra_info를 쓸 수도 있으니 더미 추가

print(f">>> 2. 리워드 계산 시도 (DataSource: {data_source})")

try:
    # 여기서 에러가 나면 진짜 원인이 출력됩니다.
    score = default_compute_score(
        data_source=data_source, 
        solution_str=solution_str, 
        ground_truth=ground_truth, 
        extra_info=extra_info
    )
    print(f">>> 3. 계산 성공! 점수: {score}")

except Exception as e:
    print("\n========================================")
    print(">>> [에러 발생] vrag.py 내부 로직 에러 가능성 높음")
    print(f"에러 메시지: {e}")
    print("========================================")
    import traceback
    traceback.print_exc()