#!/usr/bin/env python
"""Training Dynamic."""
import sys
import time
import hydra
import warnings
import os
import json

# Ignore all warnings
warnings.filterwarnings("ignore")

# Import the updated Trainer class
from src.training import Trainer

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    start_time = time.time()
    
    if sys.version_info < (3, 7, 0):
        sys.stderr.write("You need Python 3.7 or later to run this script.\n")
        sys.exit(1)
    
    trainer = Trainer(cfg)
    
    print('==== args settings ====')
    print(f'phy: {cfg.model.physical_model}; aug: {cfg.model.augmentation}; '
          f'device: {cfg.device.type}; exp_file: {trainer.outputname}; '
          f'aug_model: {cfg.model.aug_model}; gnn_graph: {cfg.neuralpde.gnn_graph}')
    print("==== end settings ====")
    
    option_dict = {
        'incomplete': 'Incomplete Param PDE',
        'complete': 'Complete Param PDE',
        'true': 'True PDE',
        'none': 'No physics',
    }
    
    print(trainer.outputname)
    print('=' * 80)
    print('#' * 80)
    print(f'# {option_dict[cfg.model.physical_model]} is used in F_p')
    print(f"# F_a is {'enabled' if cfg.model.augmentation else 'disabled'}")
    print('#' * 80)
    print('============== Big loop iteration 0 =============')
    print('=' * 80)
    
    # Train the model and get all test results
    test_results = trainer.train_leads("iter0_")
    
    # Save a summary of external test results
    if cfg.external_test.enabled:
        test_summary = {}
        for test_name, results in test_results.items():
            if test_name.startswith('external_'):
                test_summary[test_name] = {
                    'MSE': results['loss_mean'],
                    'SSE': results['loss_sum'],
                    'R_correlation': results['avg_r']
                }
        
        summary_path = os.path.join('./outputs', trainer.outputname, 'external_tests_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(test_summary, f, indent=4)
        
        print(f"\nExternal test results summary saved to: {summary_path}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("Execution time:", execution_time, "seconds")

if __name__ == "__main__":
    main()