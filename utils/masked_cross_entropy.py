import torch
from torch.nn import functional
from torch.autograd import Variable
from utils.config import *
import torch.nn as nn

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def cross_entropy(logits, target):
    batch_size = logits.size(0)
    log_probs_flat = functional.log_softmax(logits)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target)
    loss = losses_flat.sum() / batch_size
    return loss

def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))    

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1)) ## -1 means infered from other dimentions
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1)) 
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def masked_binary_cross_entropy(logits, target, length):
    '''
    logits: (batch, max_len, num_class)
    target: (batch, max_len, num_class)
    '''
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))  
    bce_criterion = nn.BCEWithLogitsLoss()
    loss = 0
    for bi in range(logits.size(0)):
        for i in range(logits.size(1)):
            if i < length[bi]:
                loss += bce_criterion(logits[bi][i], target[bi][i])
    loss = loss / length.float().sum()
    return loss


def masked_cross_entropy_(logits, target, length, take_log=False):
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))    

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1)) ## -1 means infered from other dimentions
    if take_log:
        logits_flat = torch.log(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(logits_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1)) 
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def masked_coverage_loss(coverage, attention, length):
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))    
    mask = sequence_mask(sequence_length=length) 
    min_ = torch.min(coverage, attention)
    mask = mask.unsqueeze(2).expand_as(min_)
    min_ = min_ * mask.float()
    loss = min_.sum() / (len(length)*1.0)
    return loss

def masked_cross_entropy_RL(logits, target, length, USE_CUDA=True, q=1.0):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    if (isinstance(length, torch.cuda.LongTensor) or
            isinstance(length, torch.LongTensor)):
        length = Variable(length)
    else:
        length = Variable(torch.LongTensor(length))
        if USE_CUDA:
            length = length.cuda()
    # logits_flat: (batch * max_len, num_classes)
    # -1 means infered from other dimentions
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float() * q
    loss = losses.sum() / length.float().sum()
    return loss
