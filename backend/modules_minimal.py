"""
Minimal version for YOLOv8 detection only (no segmentation, classification, or ensemble)
"""
import os
import cv2
import numpy as np

from theseus.utilities.download import download_pretrained_weights
from theseus.apis.inference.detect import DetectionPipeline  # Import directly to avoid loading other pipelines
from theseus.opt import Opts, InferenceArguments

from .edamam.api import get_info_from_db
from .constants import CACHE_DIR


class DetectionArguments:
    """Arguments for YOLOv8 detection"""
    def __init__(
        self,
        model_name: str = 'yolov8s',
        input_path: str = "",
        output_path: str = "",
        min_conf: float = 0.1,
        min_iou: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.min_conf = min_conf
        self.min_iou = min_iou
        self.tta = False
        self.tta_ensemble_mode = 'wbf'
        self.tta_conf_threshold = 0.01
        self.tta_iou_threshold = 0.9

        # Download model from cloud if needed
        if self.model_name:
            tmp_path = os.path.join(CACHE_DIR, self.model_name+'.pt')
            download_pretrained_weights(
                self.model_name,
                output=tmp_path)
            self.weight = tmp_path


def append_food_name(food_dict, class_names):
    """Append food names from labels for nutrition analysis"""
    food_labels = food_dict['labels']
    food_names = [' '.join(class_names[int(i)].split('-'))
                  for i in food_labels]
    food_dict['names'] = food_names
    return food_dict


def append_food_info(food_dict):
    """Append nutrition info from database (db.json)"""
    food_names = food_dict['names']
    food_info = get_info_from_db(food_names)
    food_dict.update(food_info)
    return food_dict


def get_prediction(
        input_path,
        output_path,
        model_name='yolov8s',
        min_iou=0.5,
        min_conf=0.1):
    """
    Simplified prediction function - YOLOv8 only, no segmentation/ensemble/TTA
    """
    
    # Read image
    ori_img = cv2.imread(input_path)
    ori_img = np.array(ori_img, dtype=np.uint16)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    # Run detection
    args = DetectionArguments(
        model_name=model_name,
        input_path=input_path,
        output_path=output_path,
        min_conf=min_conf,
        min_iou=min_iou,
    )

    det_args = InferenceArguments(key="detection")
    opts = Opts(det_args).parse_args()
    det_pipeline = DetectionPipeline(opts, args)
    class_names = det_pipeline.class_names

    result_dict = det_pipeline.inference()

    # Extract results (remove batch dimension)
    result_dict['boxes'] = result_dict['boxes'][0]
    result_dict['labels'] = result_dict['labels'][0]
    result_dict['scores'] = result_dict['scores'][0]

    # Add food names
    result_dict = append_food_name(result_dict, class_names)

    # Add nutrition info
    result_dict = append_food_info(result_dict)

    return output_path, 'detection', result_dict
