# Utilities for one cycle policy 
# The trainng testing and printing functions remain in the main notebook to allow easy modification

import torch


# OPTIMISER

def update_lr(optimizer, lr):
    """
    Updates the learning rate of the optimiser. 
    Takes the optimiser and the elarning rate.
    """
    for g in optimizer.param_groups:
        g['lr'] = lr
        
        
        
def update_mom(optimizer, mom):
    """
    Updates the momentum rate of the optimiser. 
    Takes the optimiser and the momentum.
    """
    for g in optimizer.param_groups:
        g['momentum'] = mom

        

# CHECKPOINTS

def save_checkpoint(model, is_best, filename='checkpoint.pth.tar'):
    """
    Save checkpoint if a new best is achieved
    
    Requires:
    - model
    - boolean for is_best
    - file name/path to save the model to
    """
    
    if is_best:
        torch.save(model.state_dict(), filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")
        
        
# from fastai library to load an existtin checkpoint
def load_checkpoint(model, filename = 'checkpoint.pth.tar'):
    """
    Loads an existing checkpoint. Functino from FASTAI library.
    Requires:
    - model
    - the filename/path
    """
    sd = torch.load(filename, map_location=lambda storage, loc: storage)
    names = set(model.state_dict().keys())
    for n in list(sd.keys()): 
        if n not in names and n+'_raw' in names:
            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
            del sd[n]
    model.load_state_dict(sd)

    
# Training/Testing statistics

class AvgStats(object):
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses =[]
        self.precs =[]
        self.its = []
        
    def append(self, loss, prec, it):
        self.losses.append(loss)
        self.precs.append(prec)
        self.its.append(it)
        
def accuracy(output, target, is_test=False):
    global total
    global correct
    batch_size = target.size(0)
    total += batch_size
    
    _, pred = torch.max(output, 1)
    if is_test:
        preds.extend(pred)
    correct += (pred == target).sum()
    return 100 * correct / total
    
