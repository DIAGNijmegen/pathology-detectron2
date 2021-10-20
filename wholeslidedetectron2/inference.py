# from pannukedetectron2 import labels as pannuke_labels

# from pannukedetectron2.parser import get_pannuke_coco_datadict
# from pannukedetectron2.customtrainer import register
import argparse
import csv
import os
import yaml
from pathlib import Path

# %config IPCompleter.use_jedi = False
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from wholeslidedata.iterators import create_batch_iterator

setup_logger()

inv_label_map = {
    0: "Neoplastic cells",
    1: "Inflammatory",
    2: "Connective/Soft tissue cells",
    3: "Dead Cells",
    4: "Epithelial",
    5: "Background",
}
label_map = {
    "Neoplastic cells": 0,
    "Inflammatory": 1,
    "Connective/Soft tissue cells": 2,
    "Dead Cells": 3,
    "Epithelial": 4,
    "Background": 5,
}

class Detectron2DetectionPredictor:
    def __init__(self, weights_path, output_dir, threshold):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        )

        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.SOLVER.IMS_PER_BATCH = 1

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            64  # faster, and good enough for this toy dataset (default: 512)
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_map)
        cfg.OUTPUT_DIR = output_dir
        os.makedirs(output_dir, exist_ok=True)

        cfg.MODEL.WEIGHTS = os.path.join(weights_path)  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set a custom testing threshold
        self._predictor = DefaultPredictor(cfg)

    def predict_on_batch(self, x_batch):
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = self._predictor(
            x_batch[0]
        )  

        pred_boxes = outputs["instances"].get("pred_boxes")
        scores = outputs["instances"].get("scores")
        classes = outputs["instances"].get("pred_classes")
        centers = pred_boxes.get_centers()

        predictions = []
        for idx, center in enumerate(centers):
            x, y = center.cpu().detach().numpy()
            confidence = scores[idx].cpu().detach().numpy()
            label = inv_label_map[int(classes[idx].cpu().detach())]
            prediction_record = {'x': x, 'y': y, 'label': label, 'confidence': confidence}
            predictions.append(prediction_record)
        return predictions
        

def inference(user_config, weights_path, output_dir, threshold=0.4, cpus=4):
    mode = "training"
    training_iterator = create_batch_iterator(
        mode=mode,
        user_config=user_config,
        presets=("slidingwindow",),
        cpus=cpus,
        number_of_batches=-1,
        return_info=True,
    )
    predictor = Detectron2DetectionPredictor(weights_path=weights_path, output_dir=output_dir, threshold=threshold)

    # also create json
    output_dict = {'detections': []}
    for x_batch, _, info in training_iterator:
        point = info["sample_references"][0]["point"]
        c, r = point.x, point.y
        predictions = predictor.predict_on_batch(x_batch)
        for prediction in predictions:
            x, y, label, confidence = prediction.values()
            x += c
            y += r
            prediction_record = {'x': x, 'y': y, 'label': label, 'confidence': confidence}
            output_dict.append(prediction_record)

    output_path = output_dir / 'predictions.yml'
    with open(output_path, 'w') as file:
        yaml.dump(output_dict, file)
    training_iterator.stop()


def run():
    # create argument parser
    argument_parser = argparse.ArgumentParser(description="Experiment")
    argument_parser.add_argument("--user_config", required=True)
    argument_parser.add_argument("--weights_path", required=True)
    argument_parser.add_argument("--output_dir", required=True)
    args = vars(argument_parser.parse_args())

    inference(
        user_config=args["user_config"],
        weights_path=args["weights_path"],
        output_dir=Path(args["output_dir"]),
    )


if __name__ == "__main__":
    run()
