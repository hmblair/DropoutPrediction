# __main__.py

import os 
if not os.path.exists('logs/wandb'):
    os.makedirs('logs/wandb')
os.environ['WANDB_PROJECT'] = 'DropoutPrediction'

import torch
torch.set_float32_matmul_precision('medium')

from tqdm import tqdm
tqdm._instances.clear()

from pytorch_lightning.cli import LightningCLI
if __name__ == "__main__":
    LightningCLI(
        save_config_kwargs={"overwrite": True, 'multifile' : True},
        )

"""
Example usage for fit, test, and predict:

python3 __main__.py fit \
    --config examples/fit.yaml \
    --config examples/lstm.yaml
    
python3 __main__.py test \
    --config examples/test.yaml \
    --config examples/lstm.yaml \
    --ckpt_dir checkpoints/lstm.ckpt

python3 __main__.py predict \
    --config examples/predict.yaml \
    --config examples/lstm.yaml \
    --ckpt_dir checkpoints/lstm.ckpt
"""