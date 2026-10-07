import torch
from config import Config


def load_dataset(path = 'shakespeare_complete.txt'):

    #TO DO: change into optional path
    with open(path,'r',encoding='utf-8') as f:
        text=f.read()

    #with open('input.txt','r',encoding='utf-8') as f:
    #    text=f.read()

    print ("length of dataset in characters:",len(text))

    #all the characters that would occur in the text
    chars = sorted (list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(vocab_size)

    #write vocab_size into config
    from config import Config
    Config.vocab_size = vocab_size

    #creating a mapping from characters to integers
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch  in enumerate (chars)}
    encode = lambda s:[stoi[c] for c in s] #encode:take a string, out out as a list of numbers
    decode = lambda l:''.join([itos[i] for i in l]) #decoder: take a list of numbers, out put as a string

    #encode the entire text datasetand store it into a torch.Tensor
    data = torch.tensor(encode(text), dtype=torch.long)
    print (data.shape, data.dtype)

    #seperate the data set into train and validation
    n = int(Config.train_split*len(data)) #90%train rest val
    train_data = data[:n]
    val_data = data[n:]

    return {
        "train": train_data,
        "val": val_data,
        "stoi": stoi,
        "itos": itos,
        "encode": encode,
        "decode": decode,
        "vocab_size": vocab_size,
    }