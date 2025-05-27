video_path="assets/driving_video.mp4"
image_path="assets/source_image.png"

python inference.py \
    --config config/hunyuan-portrait.yaml \
    --video_path $video_path \
    --image_path $image_path
