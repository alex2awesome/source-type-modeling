import torch.nn as nn
from typing import Optional, List, Dict, Union, Any

from torch.utils.data import Dataset
from transformers import (
    AutoModel, AutoConfig, RobertaConfig, RobertaModel, PreTrainedModel, BertPreTrainedModel,
    PretrainedConfig
)

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from batch_sequence_util import _get_attention_mask
from functools import partial

class TokenizedDataset(Dataset):
    def __init__(self, source_list=None, tokenizer=None, do_score=False, max_length=4096, label_mapper=None):
        """
        Processes a dataset, `doc_list` for either training/evaluation or scoring.
        * doc_list: We expect a list of dictionaries.
            * If `do_score=False`, then we are training/evaluating. We need a `label` field:
                [[{'sent': <sent 1>, 'label': <label 1>}, ...]]
            * If `do_score=True`, then we are scoring. We don't need a `label` field:
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels = []
        self.attention = []
        self.categories = []
        self.do_score = do_score  # whether to just score data (i.e. no labels exist)
        self.max_length = max_length
        self.label_mapper = label_mapper
        if self.label_mapper is not None:
            self.idx2label_mapper = {v:k for k,v in self.label_mapper.items()}
            self.num_labels = max(self.label_mapper.values()) + 1
        if not self.do_score:
            self.process_data(source_list)

    def transform_logits_to_labels(self, logits):
        preds = logits.argmax(dim=1)
        preds = preds.detach().cpu().numpy()
        return list(map(self.idx2label_mapper.get, preds))

    def process_one_doc(self, doc):
        sent = doc['sent'] if isinstance(doc, dict) else doc
        tokens = self.tokenizer.encode(sent)
        tokens = tokens[:self.max_length]
        return tokens

    def process_data(self, source_list):
        for doc in source_list:
            tokens = self.process_one_doc(doc)
            self.input_ids.append(tokens)
            if not self.do_score:
                self.labels.append(self.label_mapper[doc['label']])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        output_dict = {
            'input_ids': self.input_ids[idx],
        }
        if not self.do_score:
            output_dict['labels'] = self.labels[idx]
        return output_dict


def collate_fn(dataset, padding_value=0):
    """
    Takes in an instance of Torch Dataset.
    """
    # transform into dict, if it's just a list of `input_ids`
    if not isinstance(dataset[0], dict):
        dataset = list(map(lambda x: {'input_ids': x}, dataset))

    # transpose dict
    batch_by_columns = {}
    for key in dataset[0].keys():
        batch_by_columns[key] = list(map(lambda d: d[key], dataset))

    # pad input_ids
    input_ids = list(map(torch.tensor, batch_by_columns['input_ids']))
    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=padding_value
    )
    attention_mask = (input_ids != padding_value).to(int)
    output_pack = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    if 'labels' in batch_by_columns:
        output_pack['labels'] = torch.tensor(batch_by_columns['labels'])

    return output_pack


###############################
# model components
class AdditiveSelfAttention(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, input_dim)
        self.ws2 = nn.Linear(input_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ws1.state_dict()['weight'])
        self.ws1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.ws2.state_dict()['weight'])

    def forward(self, hidden_embeds, context_mask=None):
        ## get sentence encoding using additive attention (appears to be based on Bahdanau 2015) where:
        ##     score(s_t, h_i) = v_a^T tanh(W_a * [s_t; h_i]),
        ## here, s_t, h_i = word embeddings
        ## align(emb) = softmax(score(Bi-LSTM(word_emb)))
        # word_embs: shape = (num sentences in curr batch * max_len * embedding_dim)     # for word-attention:
        #     where embedding_dim = hidden_dim * 2                                       # -------------------------------------
        # sent_embs: shape = if one doc:   (num sentences in curr batch * embedding_dim)
        #         #          if many docs: (num docs x num sentences in batch x max word len x hidden_dim)
        self_attention = torch.tanh(self.ws1(self.drop(hidden_embeds)))         # self attention : if one doc: (num sentences in curr batch x max_len x hidden_dim
                                                                              #   if >1 doc: if many docs: (num docs x num sents x max word len x hidden_dim)
        self_attention = self.ws2(self.drop(self_attention)).squeeze(-1)      # self_attention : (num_sentences in curr batch x max_len)
        if context_mask is not None:
            context_mask = -10000 * (context_mask == 0).float()
            self_attention = self_attention + context_mask                    # self_attention : (num_sentences in curr batch x max_len)
        if len(self_attention.shape) == 1:
            self_attention = self_attention.unsqueeze(0)  # todo: does this cause problems?
        self_attention = self.softmax(self_attention).unsqueeze(1)            # self_attention : (num_sentences in curr batch x 1 x max_len)
        return self_attention


class AttentionCompression(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.self_attention = AdditiveSelfAttention(input_dim=hidden_size, dropout=dropout)

    def forward(self, hidden_embs, attention_mask=None):
        ## `'hidden_emds'`: shape = N x hidden_dim
        self_attention = self.self_attention(hidden_embs, attention_mask)  # self_attention = N x 1 x N
        ## batched matrix x batched matrix:
        output_encoding = torch.matmul(self_attention, hidden_embs).squeeze(1)
        return output_encoding


def freeze_hf_model(model, freeze_layers):
    def freeze_all_params(subgraph):
        for p in subgraph.parameters():
            p.requires_grad = False

    if isinstance(model, RobertaModel):
        layers = model.encoder.layer
    else:
        layers = model.transformer.h

    if freeze_layers is not None:
        for layer in freeze_layers:
            freeze_all_params(layers[layer])


class DocumentClassificationModel(PreTrainedModel):

    base_model_prefix = ''
    config_class = AutoConfig

    def __init__(self, config, hf_model=None):
        super().__init__(config)

        base_model = AutoModel.from_config(config) if hf_model is None else hf_model
        DocumentClassificationModel.base_model_prefix = base_model.base_model_prefix
        DocumentClassificationModel.config_class = base_model.config_class
        setattr(self, self.base_model_prefix, base_model)  # setattr(x, 'y', v) is equivalent to ``x.y = v''

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = config.classification_head['num_labels']
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.loss_fct = CrossEntropyLoss() if self.num_labels > 1 else BCEWithLogitsLoss()
        self.pooling_method = config.classification_head['pooling_method']
        self.word_attention = AttentionCompression(hidden_size=config.hidden_size, dropout=config.hidden_dropout_prob)
        self.post_init()

    def post_init(self):
        # during prediction, we don't have to pass this in
        if hasattr(self.config, 'freeze_layers'):
            freeze_hf_model(self.base_model, freeze_layers=self.config.freeze_layers)

    def pool_words(self, hidden, attention_mask):
        if self.pooling_method == 'average':
            return (hidden.T * attention_mask.T).T.mean(axis=1)
        elif self.pooling_method == 'cls':
            return hidden[:, 0, :]
        elif self.pooling_method == 'attention':
            return self.word_attention(hidden, attention_mask)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            sentence_lens: Optional[List[List[int]]] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Parameters:
             * `input_ids`: one document tokens (list of sentences. Each sentence is a list of ints.)
             * `labels`: list of y_preds [optional].
             * `attention`: list

        """
        if input_ids is not None and len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(dim=0)

        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pool word embeddings
        hidden = outputs[0]
        pooled_output = self.pool_words(hidden, attention_mask=attention_mask)
        # optionally, in the future we can do something like add LSTM
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # calculate loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, 1))
            else:
                loss = self.loss_fct(logits, labels)

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
