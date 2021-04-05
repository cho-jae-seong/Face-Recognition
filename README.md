# Face-detection

python api_usage/face_feature_pipeline.py #저장되어 있는 jpg파일을 읽어와 detection,alignment,crop 후에 feature를 추출하여 .npy로 저장


python api_usage/cam_util.py #cam이 연결된 상태에서 실행 시 실시간으로 face-detection과 동시에  저장된 feature와 비교하여 name 표기

# Mask를 쓴 얼굴까지도....?

Mask를 쓴 얼굴까지도 detection하여 본인의 얼굴임을 알 수 있습니다.
