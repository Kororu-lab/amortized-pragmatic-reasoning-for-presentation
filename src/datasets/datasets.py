"""
Data loading and preprocessing utilities for the amortized RSA project.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Any

from logger import get_logger

logger = get_logger(__name__)

# Import constants
from constants import (
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
)


def init_vocab(langs):
    i2w = {
        PAD_IDX: PAD_TOKEN,
        SOS_IDX: SOS_TOKEN,
        EOS_IDX: EOS_TOKEN,
        UNK_IDX: UNK_TOKEN,
    }
    w2i = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for lang in langs:
        for tok in lang:
            if tok not in w2i:
                i = len(w2i)
                w2i[tok] = i
                i2w[i] = tok
    return {'w2i': w2i, 'i2w': i2w}


def train_val_test_split(data,
                         val_size=0.1,
                         test_size=0.1,
                         random_state=None):
    """
    Split data into train, validation, and test splits
    Parameters
    ----------
    data : ``np.Array``
        Data of shape (n_data, 2), first column is ``x``, second column is ``y``
    val_size : ``float``, optional (default: 0.1)
        % to reserve for validation
    test_size : ``float``, optional (default: 0.1)
        % to reserve for test
    random_state : ``np.random.RandomState``, optional (default: None)
        If specified, random state for reproducibility
    """
    idx = np.arange(data['imgs'].shape[0])
    idx_train, idx_valtest = train_test_split(idx,
                                              test_size=val_size + test_size,
                                              random_state=random_state,
                                              shuffle=True)
    idx_val, idx_test = train_test_split(idx_valtest,
                                         test_size=test_size /
                                         (val_size + test_size),
                                         random_state=random_state,
                                         shuffle=True)
    splits = []
    for idx_split in (idx_train, idx_val, idx_test):
        splits.append({
            'imgs': data['imgs'][idx_split],
            'labels': data['labels'][idx_split],
            'langs': data['langs'][idx_split],
        })
    return splits

def load_raw_data(data_file: str) -> Dict[str, np.ndarray]:
    """
    Load and preprocess data from a .npz file.
    
    Args:
        data_file: Path to the .npz data file
        
    Returns:
        Dictionary containing 'imgs', 'labels', and 'langs' arrays
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data file is invalid or missing required keys
    """
    try:
        data = np.load(data_file)
        logger.debug(f"Loaded data file: {data_file}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading data file {data_file}: {e}")
        raise ValueError(f"Invalid data file: {data_file}") from e
    
    # Validate required keys
    required_keys = {'imgs', 'labels', 'langs'}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        raise ValueError(f"Data file missing required keys: {missing}")
    
    # Attempt to preprocess/tokenize
    # Check if langs need tokenization (are strings)
    try:
        if data['langs'].dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
            # Tokenize string data
            langs = np.array([t.lower().split() for t in data['langs']], dtype=object)
            logger.debug("Tokenized language data from strings")
        else:
            # Already tokenized
            langs = data['langs']
            logger.debug("Using pre-tokenized language data")
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error processing language data: {e}. Using raw data.")
        langs = data['langs']
    
    # Check if images need transposing
    imgs = data['imgs']
    if len(imgs.shape) == 5 and imgs.shape[-1] == 3:
        # Need to transpose from (N, K, H, W, C) to (N, K, C, H, W)
        imgs = imgs.transpose(0, 1, 4, 2, 3)
        logger.debug("Transposed image dimensions")
    
    return {
        'imgs': imgs,
        'labels': data['labels'],
        'langs': langs
    }

class ShapeWorld:
    def __init__(self, data, vocab):
        self.imgs = data['imgs']
        self.labels = data['labels']
        # Get vocab
        self.w2i = vocab['w2i']
        self.i2w = vocab['i2w']
        if len(vocab['w2i']) > 100:
            self.lang_raw = data['langs']
            self.lang_idx = data['langs']
        else:
            self.lang_raw = data['langs']
            self.lang_idx, self.lang_len = self.to_idx(self.lang_raw)

    def __len__(self):
        return len(self.lang_raw)

    def __getitem__(self, i):
        # Reference game format.
        img = self.imgs[i]
        label = self.labels[i]
        lang = self.lang_idx[i]
        return (img, label, lang)

    def to_text(self, idxs):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t) for t in langs], dtype=int) + 2
        lang_idx = np.full((len(self), max(lang_len)), self.w2i[PAD_TOKEN], dtype=int)
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[SOS_TOKEN]
            for j, tok in enumerate(toks, start=1):
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[EOS_TOKEN]
        return lang_idx, lang_len