import numpy as np
import torch
from detectron2.structures import Boxes, BoxMode, Instances
from wholeslidedata.iterators import BatchIterator


class WholeSlideDetectron2DataLoader(BatchIterator):
    def __next__(self):
        x_batch, y_batch = super().__next__()
        x_batch = x_batch / 255.0

        batch_dicts = []
        for idx, x_sample in enumerate(x_batch):
            sample_dict = {}
            target_gt_boxes = self._get_gt_boxes(y_batch[idx], x_sample.shape[:2])
            image = image.transpose(2, 0, 1).astype("float32")
            sample_dict["instances"] = target_gt_boxes
            sample_dict["image"] = torch.as_tensor(image)
            batch_dicts.append(sample_dict)
        return batch_dicts

    def _get_gt_boxes(self, y_sample, image_size):
        y_boxes = y_sample[~np.all(y_sample == 0, axis=-1)]
        boxes = [
            BoxMode.convert(obj[:4], BoxMode.XYXY_ABS, BoxMode.XYXY_ABS)
            for obj in y_boxes
        ]
        target = Instances(image_size)
        target.gt_boxes = Boxes(boxes)
        classes = [int(obj[-2]) for obj in y_boxes]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes
        return target
