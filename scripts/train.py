"""
Training script for amortized RSA models.
Supports training literal speakers (s0), literal listeners (l0), and amortized speakers.
"""
import sys
from pathlib import Path
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import contextlib
import random
from collections import defaultdict
import copy
from typing import List, Tuple
import yaml
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from vision import encoders as vision
from utils import metrics as util
from datasets.datasets import ShapeWorld
import datasets as data
from datasets.colors import ColorsInContext
from evaluation.runner import run
import models
from logger import get_logger, setup_logger
from config import Config, DataConfig

logger = get_logger(__name__)

def init_metrics():
    metrics = defaultdict(list)
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['best_epoch'] = 0
    return metrics

def generate_shapeworld_data(n_examples, n_images, data_type, img_type, output_path, generalization=None):
    """Generate ShapeWorld data if it doesn't exist."""
    output_path = Path(output_path)
    if output_path.exists():
        logger.info(f"Data file {output_path} already exists, skipping generation")
        return
    
    logger.info(f"Generating ShapeWorld data: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Call the ShapeWorld generation script
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / 'src' / 'datasets' / 'shapeworld.py'),
        '--n_examples', str(n_examples),
        '--n_images', str(n_images),
        '--data_type', data_type,
        '--img_type', img_type,
        '--out', str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to generate data: {result.stderr}")
        raise RuntimeError(f"Data generation failed for {output_path}")
    logger.info(f"Successfully generated {output_path}")

def ensure_shapeworld_data_exists(dataset_config, generalization=None):
    """Ensure all required ShapeWorld data files exist, generating them if necessary."""
    if dataset_config['name'] != 'shapeworld':
        return
    
    n_examples = dataset_config.get('n_examples', 1000)
    n_images = dataset_config.get('n_images', 3)
    data_type = dataset_config.get('data_type', 'reference')
    img_type = dataset_config.get('img_type', 'single')
    
    if generalization:
        data_dir = Path(f'./data/shapeworld/generalization/{generalization}')
        file_indices = list(range(10))  # 0-9 for generalization
    else:
        data_dir = Path('./data/shapeworld')
        # Generate all files needed for training: 0-74 (skipping 55-69)
        file_indices = list(range(75))
        file_indices = [i for i in file_indices if not (55 <= i < 70)]
    
    for idx in file_indices:
        output_path = data_dir / f'reference-{n_examples}-{idx}.npz'
        generate_shapeworld_data(n_examples, n_images, data_type, img_type, output_path, generalization)

def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def ensure_vocab_exists(args, pretrain_data):
    """Check if vocab exists, create it if not."""
    vocab_path = f'./experiments/saved_models/{args.dataset}/vocab.pt'
    
    if Path(vocab_path).exists():
        logger.info(f"Loading existing vocabulary from {vocab_path}")
        return torch.load(vocab_path)
    
    logger.info("Vocabulary not found. Generating vocabulary...")
    langs = np.array([])
    for files in pretrain_data:
        for file in files:
            d = data.load_raw_data(file)
            langs = np.append(langs, d['langs'])
    vocab = data.init_vocab(langs)
    
    # Create directory if it doesn't exist
    Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(vocab, vocab_path)
    logger.info(f"Vocabulary saved to {vocab_path}")
    return vocab

def train_literal_listeners_if_needed(args, pretrain_data, vocab):
    """Check if literal listeners exist, train them if not."""
    if args.generalization:
        listener_path = f'./experiments/saved_models/shapeworld/generalization/{args.generalization}/literal_listener_0.pt'
    else:
        listener_path = f'./experiments/saved_models/{args.dataset}/literal_listener_0.pt'
    
    if Path(listener_path).exists():
        logger.info("Literal listeners already exist. Skipping training.")
        return
    
    logger.info("Literal listeners not found. Training them now...")
    logger.info("This may take a while (~10-15 minutes)...")
    
    # Determine output files
    if args.generalization:
        output_dir = f'./experiments/saved_models/shapeworld/generalization/{args.generalization}/literal_listener_'
        output_files = [f'{output_dir}{i}.pt' for i in range(2)]
        pretrain_subset = pretrain_data[:2]  # Only first 2 for generalization
    else:
        output_dir = f'./experiments/saved_models/{args.dataset}/literal_listener_'
        if args.dataset == 'colors':
            output_files = [f'{output_dir}{i}.pt' for i in range(3)]
            pretrain_subset = pretrain_data[:3]
        else:
            output_files = [f'{output_dir}{i}.pt' for i in range(11)]
            pretrain_subset = pretrain_data
    
    # Create directory
    Path(output_files[0]).parent.mkdir(parents=True, exist_ok=True)
    
    # Train listeners
    loss = nn.CrossEntropyLoss()
    listener_lr = 0.0001
    listener_epochs = 100  # Fixed number of epochs for auto-training
    
    for idx, (file, output_file) in enumerate(zip(pretrain_subset, output_files)):
        logger.info(f"Training literal listener {idx + 1}/{len(output_files)}...")
        
        # Initialize listener
        listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
        listener_vision = vision.Conv4()
        listener = models.Listener(listener_vision, listener_embs)
        if args.cuda:
            listener = listener.cuda()
        optimizer = optim.Adam(list(listener.parameters()), lr=listener_lr)
        
        metrics = init_metrics()
        
        for epoch in range(listener_epochs):
            # Train
            data_file = file[0:len(file)-1]
            train_metrics, _ = run(data_file, 'train', 'l0', None, listener, optimizer, loss, vocab, args.batch_size, args.cuda, debug=False)
            
            # Validate
            data_file = [file[-1]]
            val_metrics, _ = run(data_file, 'val', 'l0', None, listener, optimizer, loss, vocab, args.batch_size, args.cuda, debug=False)
            
            # Track best model
            if val_metrics['acc'] > metrics['best_acc']:
                metrics['best_acc'] = val_metrics['acc']
                best_listener = copy.deepcopy(listener)
        
        # Save best listener
        torch.save(best_listener, output_file)
        logger.info(f"Saved listener to {output_file} (acc: {metrics['best_acc']:.3f})")
    
    logger.info("Finished training literal listeners!")

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Train', formatter_class=ArgumentDefaultsHelpFormatter)
    
    # Config file argument
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    
    # Dataset arguments
    parser.add_argument('--dataset', default='shapeworld', help='(shapeworld or colors)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--vocab', action='store_true', help='Generate new vocab file')
    parser.add_argument('--s0', action='store_true', help='Train literal speaker')
    parser.add_argument('--l0', action='store_true', help='Train literal listener')
    parser.add_argument('--amortized', action='store_true', help='Train amortized speaker')
    parser.add_argument('--activation', default=None)
    parser.add_argument('--penalty', default=None, help='Cost function (length)')
    parser.add_argument('--lmbd', default=0.01, help='Cost function parameter')
    parser.add_argument('--tau', default=1, type=float, help='Softmax temperature')
    parser.add_argument('--save', default='metrics.csv', help='Where to save metrics')
    parser.add_argument('--debug', action='store_true', help='Print metrics on every epoch')
    parser.add_argument('--generalization', default=None)
    parser.add_argument('--seed', default=None, type=int, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config_dict = load_config_from_yaml(args.config)
        
        # Merge config with command-line args (command-line takes precedence)
        dataset_config = config_dict.get('dataset', {})
        training_config = config_dict.get('training', {})
        model_config = config_dict.get('model', {})
        
        # Update args with config values (only if not set via command line)
        if args.dataset == 'shapeworld':  # default value
            args.dataset = dataset_config.get('name', args.dataset)
        if args.batch_size == 32:  # default value
            args.batch_size = training_config.get('batch_size', args.batch_size)
        if args.epochs == 100:  # default value
            args.epochs = training_config.get('epochs', args.epochs)
        if args.lr is None:
            args.lr = training_config.get('learning_rate', args.lr)
        if not args.cuda:
            args.cuda = training_config.get('cuda', args.cuda)
        if not args.debug:
            args.debug = training_config.get('debug', args.debug)
        if args.activation is None:
            args.activation = training_config.get('activation', args.activation)
        if args.penalty is None:
            args.penalty = training_config.get('penalty', args.penalty)
        if args.lmbd == 0.01:  # default value
            args.lmbd = training_config.get('lmbd', args.lmbd)
        if args.tau == 1:  # default value
            args.tau = training_config.get('tau', args.tau)
        if args.generalization is None:
            args.generalization = dataset_config.get('generalization', args.generalization)
        if args.seed is None:
            args.seed = training_config.get('seed', args.seed)
        
        # Set model type from config
        model_type = training_config.get('model_type', 'amortized')
        if model_type == 's0':
            args.s0 = True
        elif model_type == 'l0':
            args.l0 = True
        elif model_type == 'amortized':
            args.amortized = True
        
        # Auto-generate ShapeWorld data if needed
        logger.info("Checking if dataset files exist...")
        ensure_shapeworld_data_exists(dataset_config, args.generalization)
    
    # Set random seed if provided
    if args.seed is not None:
        logger.info(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed_all(args.seed)
    
    if args.l0 and args.lr is None:
        args.lr = 0.0001
    elif args.lr is None:
        args.lr = 0.001
    
    # Data
    if args.dataset == 'shapeworld':
        if args.generalization is None:
            data_dir = './data/shapeworld/reference-1000-'
            # Build pretrain data file lists
            pretrain_data = []
            for start in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70]:
                pretrain_data.append([f'{data_dir}{i}.npz' for i in range(start, start + 5)])
        else:
            data_dir = f'./data/shapeworld/generalization/{args.generalization}/reference-1000-'
            pretrain_data = [
                [f'{data_dir}{i}.npz' for i in range(5)],
                [f'{data_dir}{i}.npz' for i in range(5, 10)]
            ]
        train_data = [f'{data_dir}{i}.npz' for i in range(60, 65)]
        val_data = [f'{data_dir}{i}.npz' for i in range(65, 70)]
        
    elif args.dataset == 'colors':
        DatasetClass = ColorsInContext
        data_dir = './data/colors/data_1000_'
        pretrain_data = [
            [f'{data_dir}{i}.npz' for i in range(15)],
            [f'{data_dir}{i}.npz' for i in range(15, 30)],
            [f'{data_dir}{i}.npz' for i in range(30, 45)]
        ]
        train_data = [f'{data_dir}{i}.npz' for i in range(15)]
        val_data = [f'{data_dir}{i}.npz' for i in range(15, 30)]
    else:
        raise ValueError(f'Dataset {args.dataset} is not defined. Choose "shapeworld" or "colors".')
    
    # Auto-generate vocab if it doesn't exist (unless explicitly training just vocab)
    if args.vocab:
        # Force regenerate vocab
        logger.info("Regenerating vocabulary...")
        langs = np.array([])
        for files in pretrain_data:
            for file in files:
                d = data.load_raw_data(file)
                langs = np.append(langs, d['langs'])
        vocab = data.init_vocab(langs)
        vocab_path = f'./experiments/saved_models/{args.dataset}/vocab.pt'
        Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(vocab, vocab_path)
        logger.info(f"Vocabulary saved to {vocab_path}")
        return  # Exit after generating vocab
    else:
        vocab = ensure_vocab_exists(args, pretrain_data)
    
    # Initialize Speaker and Listener Model
    speaker_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    speaker_vision = vision.Conv4()
    if args.s0:
        speaker = models.LiteralSpeaker(speaker_vision, speaker_embs)
    else:
        speaker = models.Speaker(speaker_vision, speaker_embs)
    listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    listener_vision = vision.Conv4()
    listener = models.Listener(listener_vision, listener_embs)
    if args.cuda:
        speaker = speaker.cuda()
        listener = listener.cuda()
        
    # Optimization
    optimizer = optim.Adam(list(speaker.parameters())+list(listener.parameters()),lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # Initialize Metrics
    metrics = init_metrics()
    all_metrics = []

    # Pretrain Literal Listener
    if args.l0:
        if args.generalization:
            output_dir = f'./experiments/saved_models/shapeworld/generalization/{args.generalization}/literal_listener_'
            output_files = [f'{output_dir}{i}.pt' for i in range(2)]
        else:
            output_dir = f'./experiments/saved_models/{args.dataset}/literal_listener_'
            output_files = [f'{output_dir}{i}.pt' for i in range(11)]
             
        for file, output_file in zip(pretrain_data,output_files):
            # Reinitialize metrics, listener model, and optimizer
            metrics = init_metrics()
            all_metrics = []
            listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
            listener_vision = vision.Conv4()
            listener = models.Listener(listener_vision, listener_embs)
            if args.cuda:
                listener = listener.cuda()
            optimizer = optim.Adam(list(listener.parameters()),lr=args.lr)
        
            for epoch in range(args.epochs):
                # Train one epoch
                data_file = file[0:len(file)-1]
                train_metrics, _ = run(data_file, 'train', 'l0', None, listener, optimizer, loss, vocab, args.batch_size, args.cuda, debug = args.debug)

                # Validate
                data_file = [file[-1]]
                val_metrics, _ = run(data_file, 'val', 'l0', None, listener, optimizer, loss, vocab, args.batch_size, args.cuda, debug = args.debug)

                # Update metrics, prepending the split name
                for metric, value in train_metrics.items():
                    metrics['train_{}'.format(metric)].append(value)
                for metric, value in val_metrics.items():
                    metrics['val_{}'.format(metric)].append(value)
                metrics['current_epoch'] = epoch

                # Use validation accuracy to choose the best model
                is_best = val_metrics['acc'] > metrics['best_acc']
                if is_best:
                    metrics['best_acc'] = val_metrics['acc']
                    metrics['best_loss'] = val_metrics['loss']
                    metrics['best_epoch'] = epoch
                    best_listener = copy.deepcopy(listener)
                    
                if args.debug:
                    print(metrics)

            # Save the best model
            literal_listener = best_listener
            torch.save(literal_listener, output_file)
            
    # Load or train literal listeners if needed
    if args.amortized or args.s0:
        # Ensure literal listeners exist
        train_literal_listeners_if_needed(args, pretrain_data, vocab)
        
        # Load the trained listeners
        if args.generalization:
            literal_listener = torch.load(f'./experiments/saved_models/shapeworld/generalization/{args.generalization}/literal_listener_0.pt')
            literal_listener_val = torch.load(f'./experiments/saved_models/shapeworld/generalization/{args.generalization}/literal_listener_1.pt')
        else:
            literal_listener = torch.load(f'./experiments/saved_models/{args.dataset}/literal_listener_0.pt')
            literal_listener_val = torch.load(f'./experiments/saved_models/{args.dataset}/literal_listener_1.pt')

    # Train Literal Speaker
    if args.s0:
        for epoch in range(args.epochs):
            # Train one epoch
            train_metrics, _ = run(train_data, 'train', 's0', speaker, literal_listener, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, debug = args.debug)
            
            # Validate
            val_metrics, _ = run(val_data, 'val', 's0', speaker, literal_listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, debug = args.debug)
            
            # Update metrics, prepending the split name
            for metric, value in train_metrics.items():
                metrics['train_{}'.format(metric)].append(value)
            for metric, value in val_metrics.items():
                metrics['val_{}'.format(metric)].append(value)
            metrics['current_epoch'] = epoch

            # Use validation accuracy to choose the best model
            is_best = val_metrics['acc'] > metrics['best_acc']
            if is_best:
                metrics['best_acc'] = val_metrics['acc']
                metrics['best_loss'] = val_metrics['loss']
                metrics['best_epoch'] = epoch
                best_speaker = copy.deepcopy(speaker)

            if args.debug:
                print(metrics)

            # Store metrics
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.save, index=False)

        # Save the best model
        if args.generalization:
            torch.save(best_speaker, f'./experiments/saved_models/shapeworld/generalization/{args.generalization}/literal_speaker.pt')
        else:
            torch.save(best_speaker, f'./experiments/saved_models/{args.dataset}/literal_speaker.pt')
    
    # Train Amortized Speaker
    if args.amortized:
        for epoch in range(args.epochs):
            # Train one epoch
            train_metrics, _ = run(train_data, 'train', 'amortized', speaker, literal_listener, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, activation = args.activation, dataset = args.dataset, penalty = args.penalty, tau = args.tau, debug = args.debug)
            
            # Validate
            val_metrics, _ = run(val_data, 'val', 'amortized', speaker, literal_listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, activation = args.activation, dataset = args.dataset, penalty = args.penalty, tau = args.tau, debug = args.debug)

            # Update metrics, prepending the split name
            for metric, value in train_metrics.items():
                metrics['train_{}'.format(metric)].append(value)
            for metric, value in val_metrics.items():
                metrics['val_{}'.format(metric)].append(value)
            metrics['current_epoch'] = epoch
            
            # Use validation accuracy to choose the best model
            is_best = val_metrics['acc'] > metrics['best_acc']
            if is_best:
                metrics['best_acc'] = val_metrics['acc']
                metrics['best_loss'] = val_metrics['loss']
                metrics['best_epoch'] = epoch
                best_speaker = copy.deepcopy(speaker)

            if args.debug:
                print(metrics)

            # Store metrics
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.save, index=False)

        # Save the best model
        try:
            if args.generalization:
                base_path = f'./experiments/saved_models/shapeworld/generalization/{args.generalization}/'
                if args.activation == 'multinomial':
                    save_path = base_path + 'reinforce_speaker.pt'
                else:
                    save_path = base_path + 'literal_speaker.pt'
            else:
                if args.activation == 'multinomial':
                    save_path = f'./experiments/saved_models/{args.dataset}/reinforce_speaker.pt'
                elif args.penalty == 'length':
                    save_path = f'./experiments/saved_models/{args.dataset}/amortized_speaker.pt'
                else:
                    save_path = f'./experiments/saved_models/{args.dataset}/speaker.pt'
            
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_speaker, save_path)
            logger.info(f"Saved best speaker model to {save_path}")
            
        except (OSError, RuntimeError) as e:
            # Use timestamp-based fallback filename
            import time
            fallback_file = f'./experiments/saved_models/speaker_backup_{int(time.time())}.pt'
            logger.error(f"Failed to save model to intended path: {e}")
            logger.info(f"Attempting to save to fallback location: {fallback_file}")
            try:
                torch.save(best_speaker, fallback_file)
                logger.info(f"Successfully saved to {fallback_file}")
            except Exception as e2:
                logger.error(f"Failed to save model even to fallback location: {e2}")
                raise RuntimeError(f"Unable to save model: {e2}") from e