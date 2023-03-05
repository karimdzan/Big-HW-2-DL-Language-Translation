import torch
from tqdm.notebook import tqdm
from utils import create_mask
import wandb


def train_epoch(model, optimizer, train_dataloader, loss_fn, PAD_IDX, device):
    model.train()
    losses = 0

    for src, tgt in tqdm(train_dataloader, total=len(train_dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device, PAD_IDX)
        # print(src.shape)
        # print(tgt_input.shape)
        # print(tgt_padding_mask.shape)
        # print(src_padding_mask.shape)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        # print(tgt_out.shape)
        # print(logits.shape)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        wandb.log({"loss_per_batch" : loss.cpu().detach()})
        torch.cuda.empty_cache()

    return losses / len(train_dataloader)


def evaluate(model, val_dataloader, loss_fn, PAD_IDX, device):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device, PAD_IDX)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)
