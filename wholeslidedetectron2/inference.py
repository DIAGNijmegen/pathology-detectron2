# from pannukedetectron2 import labels as pannuke_labels

# from pannukedetectron2.parser import get_pannuke_coco_datadict
# from pannukedetectron2.customtrainer import register
import argparse
import csv
import os
import yaml
import json
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
import torch
from tqdm import tqdm

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

class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""

    def __call__(self, images):
        """Run d2 on a list of images.

        Args:
            images (list): BGR images of the expected shape: 720x1280
        """
        
        input_images = []
        for image in images:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            new_image = self.aug.get_transform(image).apply_image(image)
            new_image = torch.as_tensor(new_image.astype("float32").transpose(2, 0, 1))
            input_images.append({"image": new_image, "height": height, "width": width})
        
        with torch.no_grad():
            preds = self.model(input_images)
        return preds

class Detectron2DetectionPredictor:
    def __init__(self, weights_path, output_dir, threshold, nms_threshold):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        )

        cfg.DATALOADER.NUM_WORKERS = 1

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            64  # faster, and good enough for this toy dataset (default: 512)
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(label_map)
        cfg.OUTPUT_DIR = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        cfg.MODEL.WEIGHTS = os.path.join(weights_path)  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set a custom testing threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold

        self._predictor = BatchPredictor(cfg)

    def predict_on_batch(self, x_batch):
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = self._predictor(
            x_batch
        )  
        predictions = []
        for output in outputs:
            predictions.append([])
            pred_boxes = output["instances"].get("pred_boxes")
            scores = output["instances"].get("scores")
            classes = output["instances"].get("pred_classes")
            centers = pred_boxes.get_centers()
            for idx, center in enumerate(centers):
                x, y = center.cpu().detach().numpy()
                confidence = scores[idx].cpu().detach().numpy()
                label = inv_label_map[int(classes[idx].cpu().detach())]
                prediction_record = {'x': int(x), 'y': int(y), 'label': str(label), 'confidence': float(confidence)}
                predictions[-1].append(prediction_record)
        return predictions
        

def inference(user_config, weights_path, output_dir, threshold=0.0, nms_threshold=0.1, cpus=4):
    mode = "training"
    print('creating data iterator...')
    training_iterator = create_batch_iterator(
        mode=mode,
        user_config=user_config,
        presets=("folders", "slidingwindow",),
        cpus=cpus,
        number_of_batches=-1,
        return_info=True,
    )
    print('creating predictor...')
    predictor = Detectron2DetectionPredictor(weights_path=weights_path, output_dir=output_dir, threshold=threshold, nms_threshold=nms_threshold)

    # also create json
    print('predicting...')
    output_dict = {"type": 'Multiple points',
                   "version": {"major": 1,
                               "minor": 0},
                   'points': []
                  }
    for x_batch, y_batch, info in tqdm(training_iterator):
        predictions = predictor.predict_on_batch(x_batch)
        for idx, prediction in enumerate(predictions):
            point = info["sample_references"][idx]["point"]
            c, r = point.x, point.y
            for detections in prediction:
                x, y, label, confidence = detections.values()
                if label != 'Inflammatory':
                    continue

                if y_batch[idx][y][x] == 0:
                    continue

                x += c
                y += r
                prediction_record = {'point': [x,y,confidence]}
                output_dict['points'].append(prediction_record)

    print('saving predictions...')
    output_path = output_dir / 'detected-lymphocytes.json'
    with open(output_path, 'w') as outfile:
         json.dump(output_dict, outfile, indent=4)

    training_iterator.stop()
    print('finished!')


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
