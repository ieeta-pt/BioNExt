
import os
from typing import Optional, Union
from transformers import BertModel, PreTrainedModel, AutoConfig, BertModel
from transformers.modeling_outputs import  TokenClassifierOutput
from torch import nn
from torch.nn import CrossEntropyLoss

from typing import List, Optional

import torch
from itertools import islice
from .configuration_bionextextractor import BioNExtExtractorConfig


import torch

from transformers import AutoModel, PreTrainedModel, AutoConfig, BertConfig
from transformers.modeling_outputs import  TokenClassifierOutput, SequenceClassifierOutput

from torch.nn import CrossEntropyLoss
import math

class RelationLossMixin:

    def model_loss(self, logits, labels, novel=None, reduction=None):
        if reduction is None:
            return torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            return torch.nn.functional.cross_entropy(logits.view(-1,self.num_labels), labels.view(-1), reduction=reduction)

class RelationAndNovelLossMixin(RelationLossMixin):
    
    def model_loss(self, logits, labels, novel=None):
        relation_logits, novel_logits = logits
        relation_loss = super().model_loss(relation_logits, labels, reduction="none")
        novel_loss = torch.nn.functional.cross_entropy(novel_logits.view(-1, 2), novel.view(-1), reduction="none")
        per_sample_loss = relation_loss + (labels!=8).type(logits[0].dtype)*novel_loss
                                                       
        return per_sample_loss.mean()#relation_loss + (labels!=8).type(logits[0].dtype)*novel_loss(novel_logits.view(-1, 2), novel.view(-1))
        #return relation_loss + novel_loss(novel_logits.view(-1, 2), novel.view(-1))

class RelationClassifierBase(PreTrainedModel, RelationLossMixin):
    #_keys_to_ignore_on_load_unexpected = [r"pooler"]
    config_class=BioNExtExtractorConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        #print(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        
    def training_mode(self):
        if self.config.update_vocab is not None:
            self.bert.resize_token_embeddings(self.config.update_vocab)
    
    def group_embeddings_by_index(self, embeddings, indexes):
        assert len(embeddings.shape)==3
        
        batch_size = indexes.shape[0]
        max_tokens = embeddings.shape[1]
        emb_size = embeddings.shape[2]
        # masking padding
        mask_index = indexes!=-1
    
        # convert index to 1d of valid index (ignore paddings)
        indexes = indexes + mask_index*(torch.arange(batch_size).to(self.device)*max_tokens).view(batch_size,1,1)
        indexes = indexes.masked_select(mask_index)
    
        # reshape 
        embeddings = embeddings.view(batch_size*max_tokens, emb_size)
    
        # get the embeddings by index
        selected_embeddings_by_index = torch.index_select(embeddings, 0, indexes)
    
        final_output_shape = (mask_index.shape[0], mask_index.shape[1], emb_size)
        group_embeddings = torch.zeros(final_output_shape, dtype=embeddings.dtype).to(self.device).masked_scatter(mask_index, selected_embeddings_by_index)
    
        return group_embeddings, mask_index

    def classifier_representation(self, embeddings, mask = None):
        raise NotImplementedError("This is base class, pleas extend an implement classifier_representation")

    def classifier(self, class_representation, relation_mask = None):
        raise NotImplementedError("This is base class, pleas extend an implement classifier")
    
    def forward(self,
                input_ids,
            indexes=None,
            novel=None,
            labels=None,
            mask=None,
            return_dict=None,
            **model_kwargs
           ):
        # Default `model.config.use_return_dict´ is `True´
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(input_ids, return_dict=return_dict, **model_kwargs)

        assert indexes is not None

        embeddings = outputs.last_hidden_state
        
        selected_embeddings, mask_group = self.group_embeddings_by_index(embeddings, indexes)

        class_representation = self.classifier_representation(selected_embeddings, mask_group)

        logits = self.classifier(class_representation, relation_mask=mask)

        loss = None
        if labels is not None:
            loss = self.model_loss(logits, labels, novel)
        
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RelationClassifierBiLSTM(RelationClassifierBase):

    def __init__(self, config):
        super().__init__(config)
        self.num_lstm_layers = config.num_lstm_layers
        self.lstm = torch.nn.LSTM(config.hidden_size, (config.hidden_size) // 2, self.num_lstm_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(config.hidden_size, self.num_labels)  # 2 for bidirection

    def training_mode(self):
        super().training_mode()
        self.lstm.reset_parameters()
        self.fc.reset_parameters()
        
    def classifier_representation(self, embeddings, mask=None):
        out, _ = self.lstm(embeddings)
        return out[:, -1, :]
    
    def classifier(self, class_representation, mask=None):
        return self.fc(class_representation)

class RelationAndNovelClassifierBiLSTM(RelationClassifierBiLSTM, RelationAndNovelLossMixin):

    def __init__(self, config):
        super().__init__(config)
        self.fc_novel = torch.nn.Linear(config.hidden_size, 2)  # 2 for bidirection

    def training_mode(self):
        super().training_mode()
        self.fc_novel.reset_parameters()
    
    def classifier(self, class_representation):
        return super().classifier(class_representation), self.fc_novel(class_representation)

class RelationClassifierMHAttention(RelationClassifierBase):
    
    def __init__(self, config):
        super().__init__(config)

        self.weight = torch.nn.Parameter(torch.Tensor(1,1,config.hidden_size))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.MHattention_layer = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)  # 2 for bidirection
        self.fc1 = torch.nn.Linear(config.hidden_size, config.hidden_size//2)  # 2 for bidirection
        self.fc1_activation = torch.nn.GELU(approximate='none')
        self.fc2 = torch.nn.Linear(config.hidden_size//2, self.num_labels)  # 2 for bidirection

    def training_mode(self):
        super().training_mode()
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.MHattention_layer._reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
    
    def classifier_representation(self, embeddings, mask=None):
        batch_size = embeddings.shape[0]
        weight = self.weight.repeat(batch_size, 1, 1)
        
        if mask is not None:
            # flip
            mask = mask.squeeze(-1)==False
            
        out_tensors, _ = self.MHattention_layer(weight, embeddings, embeddings, key_padding_mask=mask)

        return out_tensors
    
    def classifier(self, class_representation, relation_mask = None):

        x = self.fc1(class_representation)
        x = self.fc1_activation(x) 
        logits = self.fc2(x)
        if relation_mask is not None:
            #print(logits.shape, relation_mask.shape)
            logits = logits + relation_mask.view(-1,1,self.num_labels)
        return  logits

class RelationAndNovelClassifierMHAttention(RelationClassifierMHAttention, RelationAndNovelLossMixin):
    def __init__(self, config):
        super().__init__(config)

        self.fc1_novel = torch.nn.Linear(config.hidden_size, config.hidden_size//2)  # 2 for bidirection
        self.fc1_novel_activation = torch.nn.GELU(approximate='none')
        self.fc2_novel = torch.nn.Linear(config.hidden_size//2, 2)  # 2 for bidirection

    def training_mode(self):
        super().training_mode()
        self.fc1_novel.reset_parameters()
        self.fc2_novel.reset_parameters()
    
    def classifier(self, class_representation, relation_mask=None):
        x = self.fc1_novel(class_representation)
        x = self.fc1_novel_activation(x)
        
        return super().classifier(class_representation, relation_mask=relation_mask), self.fc2_novel(x)



## Changing the name to be compatible with HF API

class BioNExtExtractorModelNoNovel(RelationClassifierMHAttention):
    config_class=BioNExtExtractorConfig

class BioNExtExtractorModel(RelationAndNovelClassifierMHAttention):
    config_class=BioNExtExtractorConfig

class BioNExtExtractorModelBiLSTMNoNovel(RelationAndNovelClassifierBiLSTM):
    config_class=BioNExtExtractorConfig

class BioNExtExtractorModelBiLSTM(RelationClassifierBiLSTM):
    config_class=BioNExtExtractorConfig

        
ARCH_MAPPING = {"mhawNovelty": BioNExtExtractorModel, 
                "mha": BioNExtExtractorModelNoNovel,
                "bilstmwNovelty" : BioNExtExtractorModelBiLSTM,
                "bilstm": BioNExtExtractorModelBiLSTMNoNovel}

def get_model_class(config):
    if config.novel:
        return ARCH_MAPPING[f"{config.arch_type}wNovelty"]
    else:
        return ARCH_MAPPING[config.arch_type]