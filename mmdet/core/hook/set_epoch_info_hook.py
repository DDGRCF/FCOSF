import os
import pickle

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch)

@HOOKS.register_module()
class SaveRecorderInfoHook(Hook):

    # def before_train_epoch(self, runner):
    #     epoch = runner.epoch
    #     model = runner.model
    #     work_dir = runner.work_dir
    #     if is_module_wrapper(model):
    #         model = model.module
    #     save_path = os.path.join(work_dir, "recorder.pkl")
    #     if os.path.exists(save_path):
    #         runner.logger.warn(f"will override {save_path}")
    #     with open(save_path, "wb") as fw:
    #         pickle.dump(getattr(model, "recorder", {}), fw)
    #     runner.logger.info(f"Epoch[{epoch}] Save to {save_path}")

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        model = runner.model
        work_dir = runner.work_dir
        if is_module_wrapper(model):
            model = model.module
        save_dir = os.path.join(work_dir, "recorder")
        save_path = os.path.join(save_dir, f"epoch_{epoch}.pkl")
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(save_path):
            runner.logger.warn(f"override {save_path}...")
        with open(save_path, "wb") as fw:
            pickle.dump(getattr(model, "recorder", {}), fw)
        runner.logger.info(f"Epoch[{epoch}] Save to {save_path}!")
        if hasattr(model, "recorder"):
            model.recorder = {}