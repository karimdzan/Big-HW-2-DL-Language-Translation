import torch
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm.notebook import tqdm
import wandb
from timeit import default_timer as timer

from train import train_epoch, evaluate
from model import Seq2SeqTransformer
from utils import set_random_seed
from data import build_vocab
from data import data_process, process_data_for_inference

set_random_seed(0xDEADF00D)

train_filepaths = ['data/train.de-en.de', 'data/train.de-en.en']
val_filepaths = ['data/val.de-en.de', 'data/val.de-en.en']
test_filepaths = ['data/test1.de-en.de']

tokenizer = get_tokenizer(None)

de_vocab = build_vocab(train_filepaths[0], tokenizer)
en_vocab = build_vocab(train_filepaths[1], tokenizer)

de_vocab.set_default_index(de_vocab['<unk>'])
en_vocab.set_default_index(en_vocab['<unk>'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch


train_data = data_process(train_filepaths, de_vocab, en_vocab, tokenizer)
val_data = data_process(val_filepaths, de_vocab, en_vocab, tokenizer)

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

SRC_VOCAB_SIZE = len(de_vocab)
TGT_VOCAB_SIZE = len(en_vocab)
EMB_SIZE = 256
NHEAD = 4
FFN_HID_DIM = 256
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

NUM_EPOCHS = 10


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


wandb.login()
wandb.init(
    project="big hw 2",
    entity='karimdzan',
    name='exp2'
)

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=generate_batch)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=generate_batch)
    train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn, PAD_IDX, device)
    end_time = timer()
    val_loss = evaluate(transformer, val_dataloader, loss_fn, PAD_IDX, device)
    if epoch % 10 == 0:
        torch.save({
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint-{epoch}-exp-2.pt')
    wandb.log({"train_loss": train_loss, "val_loss": val_loss})
    print((f"Epoch: {epoch}, "
           f"Train loss: {train_loss:.3f}, "
           f"Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
wandb.finish()


from utils import translate

test_data = process_data_for_inference('data/test1.de-en.de', de_vocab, tokenizer)

f = open("test1.de-en.en", "a+")
for sentence in tqdm(test_data):
    f.write(translate(transformer, sentence.view(-1, 1), en_vocab, BOS_IDX, EOS_IDX, device) + "\r\n")
f.close()
