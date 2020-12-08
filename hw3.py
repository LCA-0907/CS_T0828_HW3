import numpy as np
import os, json, cv2, random
from argparse import ArgumentParser

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode, Visualizer

from utils import binary_mask_to_rle

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--lr", default=0.0025, type=float)
    parser.add_argument("--thres", default=0.3, type=float)
    parser.add_argument("--iter", default=30000, type=int)
    args = parser.parse_args() 
    print("mode: ", args.mode)

    # regist my data
    register_coco_instances("my_dataset_train", {}, "pascal_train.json", "./train_images")
    register_coco_instances("my_dataset_val", {}, "test.json", "test_images")

    if args.mode == "train":
        # set cfg
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("my_dataset_train",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl" # resnet50 pretrained on ImageNet
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = args.lr # default=0.00025
        cfg.SOLVER.MAX_ITER = args.iter #default=30000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

        # train
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train() 

    elif args.mode == "test":
        # set cfg
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ()
        cfg.DATASETS.TEST = ("my_dataset_test",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thres #default=0.3

        predictor = DefaultPredictor(cfg)

        # test
        test_dir = "./test_images/"
        f = open("test.json", "r")
        test_data = json.load(f)
        test_df = []

        for img in test_data['images']:
            file_name = img['file_name']
            print(img['file_name'])
            height = img['height']
            width = img['width']
            image_id = img['id']
            im = cv2.imread(os.path.join(test_dir, file_name))
            outputs = predictor(im)
            # print(outputs, '\n')
            n_instance = len(outputs['instances'].to("cpu"))
            print(n_instance)

            for i in range(n_instance):
                pred = {}
                pred['image_id'] = image_id 
                pred['score'] = float(outputs['instances'].scores[i].to("cpu"))
                pred['category_id'] = int(outputs['instances'].pred_classes[i].to("cpu").numpy()+1)
                
                rle = binary_mask_to_rle(outputs['instances'].pred_masks[i].to("cpu").numpy())
                rle['size'] = [height, width]
                pred['segmentation'] = rle
                # print(outputs['instances'].pred_masks[:,:,i])
                test_df.append(pred)
            
        fout = open("test_out.json", "w")
        json.dump(test_df, fout)