from flask_cors import cross_origin
from flask import jsonify, request
import os

from .utils import process_upload_file
from .modules_minimal import get_prediction
from .constants import DETECTION_FOLDER


def set_routes(app):
    @app.route('/')
    def homepage():
        return jsonify({
            "message": "Food Recognition API - YOLOv8",
            "endpoint": "/analyze",
            "method": "POST",
            "parameters": "file (multipart/form-data)"
        })

    @app.route('/analyze', methods=['POST'])
    @cross_origin()
    def analyze():
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        # Process uploaded file
        filename, filepath, filetype = process_upload_file(request)
        
        if filetype != 'image':
            return jsonify({"error": "Only image files are supported"}), 400
        
        # Set output path
        output_filename = filename
        output_path = os.path.join(DETECTION_FOLDER, output_filename)
        
        # Run YOLOv8 detection
        _, _, result = get_prediction(
            input_path=filepath,
            output_path=output_path,
            model_name='yolov8s',
            min_iou=0.5,
            min_conf=0.1
        )

        boxes = result["boxes"]
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()

        scores = result["scores"]
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        labels = result["names"]  # already list/strings

        calories = result.get("calories")
        if hasattr(calories, "tolist"):
            calories = calories.tolist()

        protein = result.get("protein")
        if hasattr(protein, "tolist"):
            protein = protein.tolist()

        fat = result.get("fat")
        if hasattr(fat, "tolist"):
            fat = fat.tolist()

        carbs = result.get("carbs")
        if hasattr(carbs, "tolist"):
            carbs = carbs.tolist()

        fiber = result.get("fiber")
        if hasattr(fiber, "tolist"):
            fiber = fiber.tolist()

        return jsonify({
            "labels": labels,
            "scores": scores,
            "boxes": boxes,
            "nutrition": {
                "calories": calories,
                "protein": protein,
                "fat": fat,
                "carbs": carbs,
                "fiber": fiber,
            }
        })



    @app.after_request
    def add_header(response):
        # Include cookie for every request
        response.headers.add('Access-Control-Allow-Credentials', True)

        # Prevent the client from caching the response
        if 'Cache-Control' not in response.headers:
            response.headers['Cache-Control'] = 'public, no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '-1'
        return response