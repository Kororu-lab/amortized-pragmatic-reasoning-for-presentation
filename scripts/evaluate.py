"""
Evaluation script for amortized RSA models using config files.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
import torch
import torch.nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from vision import encoders as vision
import datasets as data
import models
from evaluation.runner import run
from logger import get_logger

logger = get_logger(__name__)

def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate trained models', formatter_class=ArgumentDefaultsHelpFormatter)
    
    # Config file or manual args
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (same as training)')
    
    # Manual arguments (for backward compatibility)
    parser.add_argument('--dataset', default='shapeworld', help='(shapeworld or colors)')
    parser.add_argument('--split', default='test', help='Data split to evaluate (train, val, test)')
    parser.add_argument('--model_type', default='amortized', help='Model type (s0, l0, amortized)')
    parser.add_argument('--speaker_path', default=None, help='Path to speaker model')
    parser.add_argument('--listener_path', default=None, help='Path to listener model')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--generalization', default=None)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config_dict = load_config_from_yaml(args.config)
        
        dataset_config = config_dict.get('dataset', {})
        training_config = config_dict.get('training', {})
        
        # Override with config values
        args.dataset = dataset_config.get('name', args.dataset)
        args.batch_size = training_config.get('batch_size', args.batch_size)
        args.cuda = training_config.get('cuda', args.cuda)
        args.debug = training_config.get('debug', args.debug)
        args.generalization = dataset_config.get('generalization', args.generalization)
        
        # Determine model type from config
        model_type = training_config.get('model_type', 'amortized')
        if model_type == 's0':
            args.model_type = 's0'
        elif model_type == 'l0':
            args.model_type = 'l0'
        elif model_type == 'amortized':
            args.model_type = 'amortized'
    
    # Determine model paths if not specified
    if args.speaker_path is None:
        model_dir = f'./experiments/saved_models/{args.dataset}'
        if args.generalization:
            model_dir = f'{model_dir}/generalization/{args.generalization}'
        
        if args.model_type == 's0':
            args.speaker_path = f'{model_dir}/literal_speaker.pt'
        elif args.model_type == 'amortized':
            args.speaker_path = f'{model_dir}/amortized_speaker.pt'
        elif args.model_type == 'reinforce':
            args.speaker_path = f'{model_dir}/reinforce_speaker.pt'
    
    if args.listener_path is None:
        model_dir = f'./experiments/saved_models/{args.dataset}'
        if args.generalization:
            model_dir = f'{model_dir}/generalization/{args.generalization}'
        args.listener_path = f'{model_dir}/literal_listener_0.pt'
    
    logger.info(f"Evaluating {args.model_type} on {args.dataset} ({args.split} split)")
    logger.info(f"Speaker: {args.speaker_path}")
    logger.info(f"Listener: {args.listener_path}")
    
    # Load vocab
    vocab_path = f'./experiments/saved_models/{args.dataset}/vocab.pt'
    vocab = torch.load(vocab_path)
    logger.info(f"Loaded vocabulary from {vocab_path}")
    
    # Load data
    if args.dataset == 'shapeworld':
        if args.generalization:
            data_dir = f'./data/shapeworld/generalization/{args.generalization}/reference-1000-'
        else:
            data_dir = './data/shapeworld/reference-1000-'
        
        if args.split == 'train':
            data_files = [f'{data_dir}{i}.npz' for i in range(60, 65)]
        elif args.split == 'val':
            data_files = [f'{data_dir}{i}.npz' for i in range(65, 70)]
        else:  # test
            data_files = [f'{data_dir}{i}.npz' for i in range(70, 75)]
    elif args.dataset == 'colors':
        data_dir = './data/colors/data_1000_'
        if args.split == 'train':
            data_files = [f'{data_dir}{i}.npz' for i in range(15)]
        elif args.split == 'val':
            data_files = [f'{data_dir}{i}.npz' for i in range(15, 30)]
        else:  # test
            data_files = [f'{data_dir}{i}.npz' for i in range(30, 45)]
    
    # Load models
    speaker_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    speaker_vision = vision.Conv4()
    if args.model_type == 's0':
        speaker = models.LiteralSpeaker(speaker_vision, speaker_embs)
    else:
        speaker = models.Speaker(speaker_vision, speaker_embs)
    
    listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    listener_vision = vision.Conv4()
    listener = models.Listener(listener_vision, listener_embs)
    
    # Load trained weights
    speaker.load_state_dict(torch.load(args.speaker_path, map_location='cpu'))
    listener.load_state_dict(torch.load(args.listener_path, map_location='cpu'))
    
    if args.cuda:
        speaker = speaker.cuda()
        listener = listener.cuda()
    
    speaker.eval()
    listener.eval()
    
    # Run evaluation
    loss = nn.CrossEntropyLoss()
    optimizer = None  # Not needed for evaluation
    
    results = run(
        data_files, 
        split='test',  # Always use test mode for evaluation
        model_type=args.model_type,
        speaker=speaker,
        listener=listener,
        optimizer=optimizer,
        loss=loss,
        vocab=vocab,
        batch_size=args.batch_size,
        cuda=args.cuda,
        dataset=args.dataset,
        generalization=args.generalization,
        debug=args.debug
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info(f"Evaluation Results ({args.dataset} - {args.split} split)")
    logger.info("=" * 60)
    
    if 'acc' in results:
        acc_vals = results['acc']
        if isinstance(acc_vals, list):
            avg_acc = sum(acc_vals) / len(acc_vals)
            logger.info(f"Accuracy: {avg_acc:.4f}")
        else:
            logger.info(f"Accuracy: {acc_vals:.4f}")
    
    if 'length' in results:
        len_vals = results['length']
        if isinstance(len_vals, list):
            avg_len = sum(len_vals) / len(len_vals)
            logger.info(f"Average Length: {avg_len:.2f}")
        else:
            logger.info(f"Average Length: {len_vals:.2f}")
    
    if 'prob' in results:
        prob_vals = results['prob']
        if isinstance(prob_vals, list):
            avg_prob = sum(prob_vals) / len(prob_vals)
            logger.info(f"Language Model Probability: {avg_prob:.4f}")
        else:
            logger.info(f"Language Model Probability: {prob_vals:.4f}")
    
    logger.info("=" * 60)

