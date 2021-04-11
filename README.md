# Face-Recognition

python api_usage/face_feature_pipeline.py ->저장되어 있는 jpg파일을 읽어와 detection,alignment,crop 후에 feature를 추출하여 .npy로 저장


python api_usage/cam_util.py ->cam이 연결된 상태에서 실행 시 실시간으로 face-detection과 동시에 저장된 feature와 비교하여 name 표기

# If you want...

원한다면 detect,alignment,crop,feature 의 과정을 따로 실행하여 결과물을 확인할 수 있습니다.

-> python api_usage/face_detect.py  # Get bounding box of a face 

-> python api_usage/face_alignment.py  # Get 106 landmark of a face 

-> python api_usage/face_crop.py  # Get croped face from a image

-> python api_usage/face_feature.py  # Get features of a face

# Mask를 쓴 얼굴까지도....?

Mask를 쓴 얼굴까지도 detection하여 본인의 얼굴임을 알 수 있습니다.


Mask를 쓴 얼굴을 처리하는 과정은 다음의 Paper를 참고하였습니다.
https://arxiv.org/abs/2101.04407
