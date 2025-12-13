UPLOAD_FOLDER = './static/assets/uploads/'
VIDEO_FOLDER = './static/assets/videos/'
CSV_FOLDER = './static/csv/'
SEGMENTATION_FOLDER = './static/assets/segmentations/'
DETECTION_FOLDER = './static/assets/detections/'
METADATA_FOLDER = './static/metadata/'

IMAGE_ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_ALLOWED_EXTENSIONS = {'mp4', 'avi', '3gpp', '3gp'}

# Use /tmp for Heroku ephemeral storage (models downloaded from cloud on-demand)
# Local development can still use ./weights if /tmp doesn't exist
import os
CACHE_DIR = '/tmp/weights' if os.environ.get('DYNO') else './weights'
