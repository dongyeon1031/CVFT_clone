import os
import cv2

# 변환할 루트 디렉토리
root_dir = "Data/CVACT/streetview"
out_dir = "Data/CVACT/streetview_png"

# 출력 폴더 없으면 생성
os.makedirs(out_dir, exist_ok=True)

# 전체 jpg/jpeg 파일 개수 세기
file_list = [f for f in os.listdir(root_dir) if f.lower().endswith((".jpg", ".jpeg"))]
total = len(file_list)

count = 0
for idx, fname in enumerate(file_list, start=1):
    in_path = os.path.join(root_dir, fname)
    out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")

    img = cv2.imread(in_path)
    if img is None:
        print(f"[WARN] 로드 실패: {in_path}")
        continue

    cv2.imwrite(out_path, img)
    count += 1

    # 진행률 표시
    print(f"[PROGRESS] {idx}/{total} ({idx/total:.2%}) 변환 완료")

print(f"[DONE] {count}/{total}개 JPG → PNG 변환 완료! 저장 경로: {out_dir}")