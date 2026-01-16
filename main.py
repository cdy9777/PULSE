from config.configurator import configs
from trainer.trainer import init_seed
from models.build_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner
import os
# Set threading limits before importing numpy, sklearn, etc.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["OMP_NUM_THREADS"] = "8"        # OpenMP (e.g., scikit-learn)
os.environ["OPENBLAS_NUM_THREADS"] = "8"   # NumPy, SciPy
os.environ["MKL_NUM_THREADS"] = "8"        # Intel MKL (used by Anaconda)
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # macOS Accelerate framework
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(4) # limit the number of threads

def main():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).to(configs['device'])

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: training
    best_model = trainer.train(model)

    # Sixth Step: test
    trainer.test(best_model)

def tune():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create logger
    logger = Logger()

    # Third Step: Create tuner
    tuner = Tuner(logger)

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)
    
    # Fifth Step: Start grid search
    tuner.grid_search(data_handler, trainer)

def test():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).to(configs['device'])

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: load model from pretrain_path
    best_model = trainer.load(model)

    # Sixth Step: test
    trainer.test(best_model)

if not configs['tune']['enable']:
    main()
else:
    tune()


