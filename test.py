import numpy as np
import torch
import torch.nn as nn

def test(model, data, mode):
    iterator = iter(getattr(data, f'{mode}_iter'))
    criterion = nn.CrossEntropyLoss()
    acc, loss, size = 0, 0, 0
    model.eval()
    with torch.set_grad_enabled(False):
        for batch in iterator:
            pred = model(batch)
            batch_loss = criterion(pred, batch.label)
            loss += batch_loss.item()
            _, pred = pred.max(dim=1)
            acc += (pred == batch.label).sum().float()
            size += len(pred)

    acc /= size
    acc = acc.cpu().item()
    return loss, acc

test()
