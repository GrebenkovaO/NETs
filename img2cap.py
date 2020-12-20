

path = '00001.jpg'
URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/%D0%9F%D0%BE%D1%80%D1%82%D1%80%D0%B5%D1%82%D0%BD%D0%BE%D0%B5_%D1%84%D0%BE%D1%82%D0%BE_%D0%91.%D0%94.%D0%9C%D0%B5%D0%BD%D0%B4%D0%B5%D0%BB%D0%B5%D0%B2%D0%B8%D1%87%D0%B0.jpg/400px-%D0%9F%D0%BE%D1%80%D1%82%D1%80%D0%B5%D1%82%D0%BD%D0%BE%D0%B5_%D1%84%D0%BE%D1%82%D0%BE_%D0%91.%D0%94.%D0%9C%D0%B5%D0%BD%D0%B4%D0%B5%D0%BB%D0%B5%D0%B2%D0%B8%D1%87%D0%B0.jpg'



import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import torch, torch.nn as nn
import torch.nn.functional as F
import pickle
from torchvision.datasets.folder import default_loader
from typing import List
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from warnings import warn
from torchvision.models.vgg import VGG, cfgs


with open('vocab.bin', 'rb') as fin:
    vocab = pickle.load(fin)
with open('word_to_index.bin', 'rb') as fin:
    word_to_index = pickle.load(fin)

eos_ix = word_to_index['#END#']
unk_ix = word_to_index['#UNK#']
pad_ix = word_to_index['#PAD#']

def as_matrix(sequences, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    max_len = max_len or max(map(len,sequences))
    
    matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad_ix
    for i,seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    
    return matrix


# In[46]:


class ScaledDotProductScore(nn.Module):
    """
    Vaswani et al. "Attention Is All You Need", 2017.
    """
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys):
        """
        queries:  [batch_size x num_queries x dim]
        keys:     [batch_size x num_objects x dim]
        Returns a tensor of scores with shape [batch_size x num_queries x num_objects].
        """
        result = torch.bmm(queries, keys.permute(0, 2, 1)) / (keys.shape[2]**0.5)
        return result


# In[47]:


class Attention(nn.Module):
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, queries, keys, values):
        """
        queries:         [batch_size x num_queries x query_feature_dim]
        keys:            [batch_size x num_objects x key_feature_dim]
        values:          [batch_size x num_objects x obj_feature_dim]
        Returns matrix of responses for queries with shape [batch_size x num_queries x obj_feature_dim].
        Saves detached weights as self.attention_map.
        """
        scores = self.scorer(queries, keys)
        weights = F.softmax(scores, dim=2) 
        self.attention_map = weights.detach()
        result = torch.bmm(weights, values)
        return result


# In[48]:


class CaptionNet(nn.Module):
    def __init__(self, n_tokens=0, emb_size=128, lstm_units=256, cnn_channels=512):
        """ A recurrent 'head' network for image captioning. Read scheme below. """
        super(self.__class__, self).__init__()
        
        # a layer that converts conv features to 
        self.cnn_to_h0 = nn.Linear(cnn_channels, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_channels, lstm_units)
        
        # recurrent part, please create the layers as per scheme above.

        # create embedding for input words. Use the parameters (e.g. emb_size).
        self.emb = nn.Embedding(n_tokens, emb_size, max_norm=5)
            
        # attention: create attention over image spatial positions
        # The query is previous lstm hidden state, the keys are transformed cnn features,
        # the values are cnn features
        self.attention = Attention(ScaledDotProductScore())
        
        # attention: create transform from cnn features to the keys
        # Hint: one linear layer shoud work
        # Hint: the dimensionality of keys should be lstm_units as lstm
        #       hidden state is the attention query
        self.cnn_to_attn_key = nn.Linear(cnn_channels, lstm_units)
                
        # lstm: create a recurrent core of your network. Use LSTMCell
        self.lstm = nn.LSTMCell(cnn_channels+emb_size, lstm_units)

        # create logits: MLP that takes attention response, lstm hidden state
        # and the previous word embedding as an input and computes one number per token
        # Hint: I used an architecture with one hidden layer, but you may try deeper ones
        self.logits_mlp = nn.Linear(cnn_channels+emb_size+lstm_units, n_tokens)
        
    def forward(self, image_features, captions_ix):
        """ 
        Apply the network in training mode. 
        :param image_features: torch tensor containing VGG features for each position.
                               shape: [batch, cnn_channels, width * height]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i]. 
            padded with pad_ix
        :returns: logits for next token at each tick, shape: [batch, word_i, n_tokens]
        """
        initial_cell = self.cnn_to_c0(image_features.mean(2))
        initial_hid = self.cnn_to_h0(image_features.mean(2))
        
        image_features = image_features.transpose(1, 2)
        
        # compute embeddings for captions_ix
        captions_emb = self.emb(captions_ix)
        
        reccurent_out = []
        attention_map = []
        
        cell = initial_cell
        hid = initial_hid
        key = self.cnn_to_attn_key(image_features)
        for i in range(captions_ix.shape[1]):
            a = self.attention(hid.view(len(key), 1, -1), key, image_features).view(len(key), -1)
            attention_map.append(self.attention.attention_map)
            hid, cell = self.lstm(torch.cat((captions_emb[:, i], a), dim=1), (hid, cell))
            reccurent_out.append(torch.cat((hid, a, captions_emb[:, i]), dim = 1).view(len(key), 1, -1))
        reccurent_out = torch.cat(reccurent_out, dim=1)
        attention_map = torch.cat(attention_map, dim=1)
        
        # compute logits for next token probabilities
        # based on the stored in (2.7) values (reccurent_out)
        bs, cl, dim = reccurent_out.shape
        logits = self.logits_mlp(reccurent_out.view(-1, dim)).view(bs, cl, -1)
        
        # return logits and attention maps from (2.4)
        return logits, attention_map


# In[49]:


class BeheadedVGG19(VGG):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """
    
    def forward(self, x):
        x_for_attn = x= self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = x = self.classifier(x)
        return x_for_attn, logits




def generate_caption(features_net, network, image, caption_prefix = ("#START#",), 
                     t=1, sample=True, max_len=100):
    
    assert isinstance(image, np.ndarray) and np.max(image) <= 1           and np.min(image) >=0 and image.shape[-1] == 3
    
    image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)
    
    vectors_9x9, logits = features_net(image[None])
    caption_prefix = list(caption_prefix)
    
    attention_maps = []
    
    for _ in range(max_len):
        
        prefix_ix = as_matrix([caption_prefix])
        prefix_ix = torch.tensor(prefix_ix, dtype=torch.int64)
        input_features = vectors_9x9.view(vectors_9x9.shape[0], vectors_9x9.shape[1], -1)
        if next(network.parameters()).is_cuda:
            input_features, prefix_ix = input_features.cuda(), prefix_ix.cuda()
        else:
            input_features, prefix_ix = input_features.cpu(), prefix_ix.cpu()
        next_word_logits, cur_attention_map = network(input_features, prefix_ix)
        next_word_logits = next_word_logits[0, -1]
        cur_attention_map = cur_attention_map[0, -1]
        next_word_probs = F.softmax(next_word_logits, -1).detach().cpu().numpy()
        attention_maps.append(cur_attention_map.detach().cpu())
        
        assert len(next_word_probs.shape) ==1, 'probs must be one-dimensional'
        next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t) # apply temperature

        if sample:
            next_word = np.random.choice(vocab, p=next_word_probs) 
        else:
            next_word = vocab[np.argmax(next_word_probs)]

        caption_prefix.append(next_word)

        if next_word=="#END#":
            break

    return caption_prefix[1:-1]


def capt(network, features_net, path='0', URL=''):
        
    if URL!='':
        # from URL
        response = requests.get(URL)
        img = Image.open(BytesIO(response.content))
        img.save('./images/input.jpg')
    else:
        # from file
        img = default_loader(path)
	
    img = np.array(img)
    img = img/img.max()
    return generate_caption(features_net, network, img, t=5)

