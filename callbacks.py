# callbacks.py
from transformers import TrainerCallback

class TrainingMonitorCallback(TrainerCallback):
    """A callback to monitor training progress."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(f"Step {state.global_step}:")
            if "loss" in logs:
                print(f"  Loss: {logs['loss']:.4f}")
            if "eval_loss" in logs:
                print(f"  Eval Loss: {logs['eval_loss']:.4f}")

    def on_init_end(self, args, state, control, **kwargs):
        print("Training initialization complete.")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} complete.")
