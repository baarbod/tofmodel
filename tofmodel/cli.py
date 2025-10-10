# -*- coding: utf-8 -*-

import argparse
from omegaconf import OmegaConf
import os


def main():
    
    # DEFINE PARSERS
    parser = argparse.ArgumentParser(description="TOF Model Framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    view_parser = subparsers.add_parser("view", help="view things before or after the inverse model pipeline")
    view_parser.add_argument("--config", type=str, required=True, help="path to config yml file")
    view_parser.add_argument("--output", type=str, help="output directory")
    view_parser.add_argument("--component", type=str, choices=["sampling", "simulations"], required=True,
                            help="Choose what to view")
    
    inverse_parser = subparsers.add_parser("inverse", help="run the inverse model pipeline")
    inverse_parser.add_argument("--config", type=str, required=True, help="path to config yml file")
    inverse_parser.add_argument("--mode", type=str, choices=['singletask', 'sequential'])
    inverse_parser.add_argument("--taskid", type=int, help="Batch/task ID (1-based)")
    inverse_parser.add_argument("--action", type=str,
                        choices=[
                            "prepare_inputs", "sort_inputs", "run_simulations", 
                            "combine_simulations", "cleanup_directories", "run_all"
                        ], help="Operation to perform")
    
    train_parser = subparsers.add_parser("train", help="train the model on the simulated dataset")
    train_parser.add_argument("--dataset", type=str, required=True, help="path to dataset file")
    train_parser.add_argument("--noisedir", type=str, required=True, help="path to directory where noise files are")
    train_parser.add_argument("--epochs", type=int, required=True, help="number of training epochs")
    train_parser.add_argument("--batch", type=int, required=True, help="batch size")
    train_parser.add_argument("--lr", type=float, required=True, help="learning rate for optimizer")
    train_parser.add_argument("--noise_method", type=str, default='none', required=False, help="method of noise injection")
    train_parser.add_argument("--gauss_low", type=float, required=False, help="lower bound for gaussian noise sampling")
    train_parser.add_argument("--gauss_high", type=float, required=False, help="upper bound for gaussian noise sampling")
    train_parser.add_argument("--noise_scale", type=float, required=False, help="scale factor for pca-based noise sampling")
    train_parser.add_argument("--exp_name", type=str, required=False, help="name of experiment (appended to formatted name)")
    
    # PARSE
    args = parser.parse_args()

    # PERFORM OPERATIONS
    if args.command == "view":
        run_view(args)
    
    if args.command == 'train':
        run_train(args)

    if args.command == "inverse":
        
        # VALIDATE INPUT ARGS LOGIC
        
        # default to sequential for run_all if mode not provided
        if args.action == "run_all":
            if args.mode is None:
                args.mode = "sequential"
            elif args.mode != "sequential":
                parser.error("--mode must be 'sequential' for action 'run_all'. Either specify sequential or do not specify")
                
        # mode is required for actions using multithreading
        if args.action in ["prepare_inputs", "run_simulations"]:
            if args.mode is None:
                parser.error(f"--mode must be specified with action '{args.action}'")

        # mode/taskid are only used for actions using multithreading
        if args.action in ["sort_inputs", "combine_simulations", "cleanup_directories"]:
            if args.mode is not None or args.taskid is not None:
                parser.error(f"--mode and/or --taskid cannot be used with action '{args.action}'")
                    
        elif args.mode not in ["singletask"] and args.taskid is not None:
            parser.error("--taskid is only valid when --mode is 'singletask'")
                
        run_inverse(args)
        

def run_view(args):
    
    if args.component == "sampling":
        from tofmodel.inverse.view import view_sampling
        view_sampling(args.config)
        
    elif args.component == "simulations":
        from tofmodel.inverse.view import view_simulations
        view_simulations(args.config)


def run_train(args):
    from tofmodel.inverse import train
    train.train_net(args.dataset, args.noisedir, epochs=args.epochs, batch=args.batch, lr=args.lr, noise_method=args.noise_method, 
              gauss_low=args.gauss_low, gauss_high=args.gauss_high, noise_scale=args.noise_scale, exp_name=args.exp_name)
    
    
def print_message(action, mode, taskid, param):
    
    print('========================================')
    print(f"Running {action} using {mode} mode.")
    print(f"Current taskid: {taskid}")
    print(f"Total batches: {param.data_simulation.num_batches}")
    print(f"Total samples: {param.data_simulation.num_samples}")
    print(f"Current batch size: {param.data_simulation.num_samples // param.data_simulation.num_batches}")
    print('========================================')
   
    
def run_inverse(args):
    param = load_config(args.config)

    if (
        param.scan_param.num_pulse != param.data_simulation.input_feature_size
        or param.scan_param.num_pulse != param.data_simulation.output_feature_size
    ):
        raise ValueError(
            f"num_pulse ({param.scan_param.num_pulse}) must match both "
            f"input ({param.data_simulation.input_feature_size}) and output ({param.data_simulation.output_feature_size}) feature sizes"
        )

    dirs = setup_directories(param)
    action = args.action
    mode = args.mode

    def _prepare_inputs():
        from tofmodel.inverse.dataset import prepare_inputs
        if mode == 'sequential':
            for taskid in range(1, param.data_simulation.num_batches + 1):
                print_message("prepare_inputs", mode, taskid, param)
                prepare_inputs(param, dirs, taskid)
        elif mode == 'singletask':
            print_message("prepare_inputs", mode, args.taskid, param)
            prepare_inputs(param, dirs, args.taskid)

    def _sort_inputs():
        from tofmodel.inverse.dataset import sort_inputs
        ds_param = param.data_simulation
        batch_size = ds_param.num_samples // ds_param.num_batches
        sort_inputs(dirs['batched'], dirs['sorted'], batch_size)

    def _run_simulations():
        from tofmodel.inverse.dataset import run_simulations
        if mode == 'sequential':
            for taskid in range(1, param.data_simulation.num_batches + 1):
                print_message("run_simulations", mode, taskid, param)
                run_simulations(param, dirs, taskid)
        elif mode == 'singletask':
            print_message("run_simulations", mode, args.taskid, param)
            run_simulations(param, dirs, args.taskid)

    def _combine_simulations():
        from tofmodel.inverse.dataset import combine_simulations
        combine_simulations(param, dirs)

    def _cleanup_directories():
        from tofmodel.inverse.dataset import cleanup_directories
        cleanup_directories(dirs)

    if action == 'run_all':
        if mode is None:
            raise ValueError("--mode must be specified with action 'run_all'")
        _prepare_inputs()
        _sort_inputs()
        _run_simulations()
        _combine_simulations()
        _cleanup_directories()

    elif action == 'prepare_inputs':
        if mode is None:
            raise ValueError("--mode must be specified with action 'prepare_inputs'")
        _prepare_inputs()

    elif action == 'sort_inputs':
        if mode is not None or args.taskid is not None:
            raise ValueError("--mode and/or --taskid cannot be used with action 'sort_inputs'")
        _sort_inputs()

    elif action == 'run_simulations':
        if mode is None:
            raise ValueError("--mode must be specified with action 'run_simulations'")
        _run_simulations()

    elif action == 'combine_simulations':
        if mode is not None or args.taskid is not None:
            raise ValueError("--mode and/or --taskid cannot be used with action 'combine_simulations'")
        _combine_simulations()

    elif action == 'cleanup_directories':
        if mode is not None or args.taskid is not None:
            raise ValueError("--mode and/or --taskid cannot be used with action 'cleanup_directories'")
        _cleanup_directories()

    else:
        raise ValueError(f"Unknown action: {action}")


def load_config(path):
    return OmegaConf.load(path)


def setup_directories(param):
    output_dir = param.output_dir
    dataset_name = param.dataset_name
    datasetdir = os.path.join(output_dir, dataset_name)
    dirs = {
        'batched': os.path.join(datasetdir, 'inputs_batched'),
        'sorted': os.path.join(datasetdir, 'inputs_batched_sorted'),
        'sim_batched': os.path.join(datasetdir, 'simulations_batched'),
        'dataset': datasetdir,
        'data': os.path.join(output_dir, 'data')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


if __name__ == "__main__":
    main()