from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.events import EventStorage
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import cv2
import torch.multiprocessing
from torch import multiprocessing
import torch.multiprocessing as mp
from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import sys

torch.multiprocessing.set_sharing_strategy('file_system')

# Register your dataset
register_coco_instances("my_dataset_train", {}, r"D:\ResearchProject_SyedZaidi\AI_Images\train\coco_annotations.json",
                        r"D:\ResearchProject_SyedZaidi\AI_Images\train\Images")
register_coco_instances("my_dataset_val", {}, r"D:\ResearchProject_SyedZaidi\RealImages\test\RGBImages\coco.json",
                        r"D:\ResearchProject_SyedZaidi\RealImages\test\RGBImages")


# Set up the configuration
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.PIN_MEMORY = True
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 20700  # Ensure the total number of iterations is large enough
    cfg.SOLVER.STEPS = (4000, 5000)  # Set the learning rate decay steps within the MAX_ITER
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.OUTPUT_DIR = r"D:\ResearchProject_SyedZaidi\carla-simulator-data-model-training\Faster R-CNN\OutputTrainAITestReal_2\output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


class CustomTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_ap50 = 0.0
        self.no_improve_count = 0
        self.patience = 10
        self.stop_training = False  # Flag to indicate early stopping

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    def train(self):
        """
        Train the model with proper resumption from the best checkpoint.
        """
        # Load the best model checkpoint
        best_model_path = os.path.join(self.cfg.OUTPUT_DIR, "model_best.pth")
        if os.path.exists(best_model_path):
            print(f"Resuming from best model checkpoint: {best_model_path}")
            checkpoint_data = torch.load(best_model_path)
            self.checkpointer.model.load_state_dict(checkpoint_data["model_state"])
            self.optimizer.load_state_dict(checkpoint_data["optimizer_state"])
            self.start_iter = checkpoint_data.get("iteration", 0)
            self.iter = self.start_iter  # Update the trainer's iteration
            self.best_ap50 = checkpoint_data.get("best_ap50", 0.0)
            self.no_improve_count = checkpoint_data.get("no_improve_count", 0)
        elif self.checkpointer.has_checkpoint():
            # Fallback to last checkpoint if best checkpoint is not available
            print("Resuming from the last saved checkpoint.")
            self.checkpointer.load(self.cfg.MODEL.WEIGHTS)
        else:
            print("No checkpoint found. Starting fresh training.")

        # Adjust max iterations if resuming from an intermediate point
        remaining_iters = self.cfg.SOLVER.MAX_ITER - self.start_iter
        self.scheduler = self.build_lr_scheduler(self.cfg, optimizer=self.optimizer)
        print(f"Starting training from iteration {self.start_iter} with {remaining_iters} remaining iterations.")

        # Initialize the data loader iterator for the training data
        self._data_loader_iter = iter(self.data_loader)

        # Set EventStorage to match the current iteration
        with EventStorage(self.start_iter) as storage:
            super().train()

    def run_step(self):
        if self.stop_training:
            print("Stopping training as early stopping was triggered.")
            sys.exit(0)  # Exit the training loop

        assert self.model.training, "[CustomTrainer] model was not in training mode!"
        with EventStorage(self.iter) as storage:
            try:
                data = next(self._data_loader_iter)
            except StopIteration:
                self._data_loader_iter = iter(self.data_loader)
                data = next(self._data_loader_iter)

            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            if torch.isnan(losses).any():
                raise ValueError(f"Loss is NaN at iteration {self.iter}")
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            storage.put_scalars(total_loss=losses, **loss_dict)
            print(f"Iteration {self.iter}: Total Loss = {losses:.4f}")

            if (self.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
                self.perform_evaluation()


    def perform_evaluation(self):
        val_loader = build_detection_test_loader(self.cfg, "my_dataset_val")
        evaluator = self.build_evaluator(self.cfg, "my_dataset_val")
        results = inference_on_dataset(self.model, val_loader, evaluator)

        ap50 = results["bbox"]["AP50"]
        print(f"AP50 at iteration {self.iter}: {ap50}")

        if ap50 > self.best_ap50:
            self.best_ap50 = ap50
            self.no_improve_count = 0
            # Save the best model
            best_model_path = os.path.join(self.cfg.OUTPUT_DIR, "model_best.pth")
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "iteration": self.iter,
                    "best_ap50": self.best_ap50,
                    "no_improve_count": self.no_improve_count,
                },
                best_model_path,
            )
            print(f"New best model saved at iteration {self.iter}: {best_model_path}")
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            print(f"Early stopping triggered at iteration {self.iter}. Exiting training.")
            self.stop_training = True  # Set the flag to stop further training


# Save detected objects as images
def save_detected_objects(cfg):
    output_dir = r"D:\ResearchProject_SyedZaidi\carla-simulator-data-model-training\Faster R-CNN\OutputTrainAITestReal_2\Output_detectedimages"
    os.makedirs(output_dir, exist_ok=True)

    # Load the checkpoint and extract model weights
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_weights = checkpoint["model_state"]

    # Save weights only for compatibility with DefaultPredictor
    weights_only_path = os.path.join(cfg.OUTPUT_DIR, "model_weights_only.pth")
    torch.save(model_weights, weights_only_path)

    # Use the extracted weights for prediction
    cfg.MODEL.WEIGHTS = weights_only_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # Set detection threshold
    predictor = DefaultPredictor(cfg)

    # Load the validation dataset
    dataset_dicts = DatasetCatalog.get("my_dataset_val")

    # Visualize and save detected results
    for idx, d in enumerate(dataset_dicts):
        img = cv2.imread(d["file_name"])  # Load the image
        outputs = predictor(img)  # Get predictions

        # Visualize predictions on the image
        v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("my_dataset_val"), scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Save the visualized image
        save_path = os.path.join(output_dir, f"detected_{os.path.basename(d['file_name'])}")
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])

        print(f"Saved: {save_path}")


# Main function
def main():
    cfg = setup_cfg()

    # Load best checkpoint if available and update max iterations
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
    if os.path.exists(best_model_path):
        print(f"Using best model checkpoint: {best_model_path}")
        cfg.MODEL.WEIGHTS = best_model_path
        checkpoint_data = torch.load(best_model_path)
        last_iteration = checkpoint_data.get("iteration", 0)
        cfg.SOLVER.MAX_ITER = max(cfg.SOLVER.MAX_ITER, last_iteration)
    else:
        print("Best model checkpoint not found. Using default weights.")

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)

    try:
        trainer.train()
    finally:
        # Ensure images are saved after training completes or stops
        save_detected_objects(cfg)
        print("Image saving completed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    mp.set_start_method('spawn', force=True)
    main()
