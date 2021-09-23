from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from wholeslidedata.iterators import create_batch_iterator

from wholeslidedetectron2.dataloader import WholeSlideDetectron2DataLoader


class WholeSlideDectectron2Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        user_config = "./configs/detection_config.yml"
        cpus = 1
        mode = "training"

        training_batch_generator = create_batch_iterator(
            user_config=user_config,
            mode=mode,
            cpus=cpus,
            iterator_class=WholeSlideDetectron2DataLoader,
        )
        return training_batch_generator


def train():
    #     coco_datadict = get_pannuke_coco_datadict(data_folder, fold)
    #     register(fold, coco_datadict)
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("detection_dataset2",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    #     cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = None
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        64  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = "/home/user/output/"
    cfg.SOLVER.STEPS = (1000, 10000, 20000, 50000, 100000)
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.GAMMA = 0.5
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = WholeSlideDataDetectionTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def collect_arguments():
    return {}


if __name__ == "__main__":
    args = collect_arguments()
    train()
