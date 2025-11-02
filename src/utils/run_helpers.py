"""
Helper functions for the main training and evaluation runner.
"""
import numpy as np
import torch
import torch.nn.functional as F

from constants import COLOR_TOKENS, SHAPE_TOKENS, SOS_IDX, EOS_IDX, PAD_IDX

def compute_average_metrics(meters):
    """Computes and returns a dictionary of average metrics."""
    metrics = {m: vs.avg for m, vs in meters.items()}
    metrics = {
        m: v if isinstance(v, float) else v.item()
        for m, v in metrics.items()
    }
    return metrics

def collect_outputs(meters, outputs, vocab, img, y, lang, lang_length, lis_pred, lis_scores, this_loss, this_acc, batch_size, ci_listeners, language_model, times):
    """Collects batch outputs and updates metrics for evaluation."""
    seq_prob = []
    if language_model:
        probs = language_model.probability(lang, lang_length).cpu().numpy()
        for prob in probs:
            seq_prob.append(np.exp(prob))

    ci = []
    if ci_listeners:
        for ci_listener in ci_listeners:
            correct = (ci_listener(img, lang, lang_length).argmax(1) == y)
            acc = correct.float().mean().item()
            ci.append(acc)

    lang_indices = lang.argmax(2)
    outputs['lang'].append(lang_indices)
    outputs['pred'].append(lis_pred)
    outputs['score'].append(lis_scores)
    meters['loss'].append(this_loss.cpu().numpy())
    meters['acc'].append(this_acc)
    if seq_prob:
        meters['prob'].append(seq_prob)
    if ci:
        meters['ci_acc'].append(ci)
    
    # -2 to account for SOS and EOS tokens
    meters['length'].append(lang_length.cpu().numpy() - 2)

    # Count color and shape tokens
    colors = sum((lang_indices == color).sum(dim=1).float() for color in COLOR_TOKENS)
    shapes = sum((lang_indices == shape).sum(dim=1).float() for shape in SHAPE_TOKENS)
    
    meters['colors'].append(colors.cpu().numpy())
    meters['shapes'].append(shapes.cpu().numpy())
    meters['time'].append(times)
    
    return meters, outputs

def generate_utterance(token_1, token_2, batch_size, max_len, vocab, device='cpu'):
    """Generate a fixed utterance with specific tokens for evaluation."""
    vocab_size = len(vocab['w2i'])
    lang = torch.zeros(batch_size, max_len, vocab_size, device=device)
    lang_length = torch.zeros(batch_size, device=device)

    lang[:, 0, SOS_IDX] = 1
    lang[:, 1, token_1] = 1
    
    if token_2 is not None:
        lang[:, 2, token_2] = 1
        lang[:, 3, EOS_IDX] = 1
        lang[:, 4:, PAD_IDX] = 1
        lang_length.fill_(4)
    else:
        lang[:, 2, EOS_IDX] = 1
        lang[:, 3:, PAD_IDX] = 1
        lang_length.fill_(3)
        
    return lang.unsqueeze(0), lang_length.unsqueeze(0)

def select_best_utterance(langs, lang_lengths, internal_listener, img, y, alpha=1.0):
    """Selects the best utterance from a set of candidates based on listener score."""
    import math
    best_score_diff = -math.inf * torch.ones(langs.shape[1], device=img.device)
    best_lang = torch.zeros_like(langs[0])
    best_lang_length = torch.zeros_like(lang_lengths[0])

    for lang, lang_length in zip(langs, lang_lengths):
        with torch.no_grad():
            lis_scores = internal_listener(img, lang, lang_length)
        
        # Score is difference between target prob and mean distractor prob
        target_scores = lis_scores.gather(1, y.unsqueeze(1)).squeeze()
        distractor_mask = torch.ones_like(lis_scores).bool()
        distractor_mask.scatter_(1, y.unsqueeze(1), False)
        distractor_scores = lis_scores[distractor_mask].view(lis_scores.size(0), -1).mean(dim=1)
        
        score_diff = target_scores - distractor_scores
        penalized_score = score_diff - alpha * lang_length.float()

        # Update best utterance for each item in the batch
        is_better = penalized_score > best_score_diff
        best_score_diff[is_better] = penalized_score[is_better]
        best_lang[is_better] = lang[is_better]
        best_lang_length[is_better] = lang_length[is_better]

    return best_lang, best_lang_length

def format_language_sequence(seq, vocab):
    """Converts a sequence of token indices to a human-readable string."""
    return ' '.join(vocab['i2w'].get(idx, '<UNK>') for idx in seq)

def prepare_language_input(lang, vocab_size, max_len, cuda):
    """Prepares language tensor for model input (one-hot encoding, padding)."""
    length = torch.tensor([np.count_nonzero(t) for t in lang.cpu()], dtype=torch.long)
    lang[lang >= vocab_size] = PAD_IDX  # Use PAD for out-of-vocab indices
    lang = F.one_hot(lang, num_classes=vocab_size)
    lang = F.pad(lang, (0, 0, 0, max_len - lang.shape[1])).float()
    
    if cuda:
        lang = lang.cuda()
        length = length.cuda()
        
    return lang, length

def extract_language_length(lang_onehot, eos_idx):
    """Calculates the length of sequences in a one-hot tensor."""
    lengths = []
    for seq in lang_onehot.cpu().numpy():
        eos_positions = np.where(seq == eos_idx)[0]
        length = eos_positions[0] + 1 if len(eos_positions) > 0 else len(seq)
        lengths.append(length)
    return torch.tensor(lengths, device=lang_onehot.device)