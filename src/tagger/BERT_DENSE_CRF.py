#from transformers import BertPreTrainedModel, BertForSequenceClassification, BertModel
import os
from typing import Optional, Union
from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import  TokenClassifierOutput
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from layers.CRF import CRF
from itertools import islice


NUM_PER_LAYER = 16

class BERTDenseCRF(PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained(config._name_or_path, config=config, add_pooling_layer=False)
        # self.vocab_size = config.vocab_size
        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_activation = nn.GELU(approximate='none')
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.reduction=config.crf_reduction
        
        if self.config.freeze == True:
            self.manage_freezing()
        
        #self.bert.init_weights() # load pretrained weights
    
    def manage_freezing(self):
        for _, param in self.bert.embeddings.named_parameters():
            param.requires_grad = False
        
        num_encoders_to_freeze = self.config.num_frozen_encoder
        if num_encoders_to_freeze > 0:
            for _, param in islice(self.bert.encoder.named_parameters(), num_encoders_to_freeze*NUM_PER_LAYER):
                param.requires_grad = False
    
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
               ):
        # Default `model.config.use_return_dict´ is `True´
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # B S E 
        dense_output = self.dense(sequence_output)
        dense_output = self.dense_activation(dense_output)
        logits = self.classifier(dense_output)
        #logits = self.classifier(sequence_output)

        loss = None
        if labels is not None: 
            # During train/test as we don't pass labels during inference
            
            # loss
            return self.crf(logits, labels, reduction=self.reduction), logits
        else:
            # decoded tags
            # NOTE: This gather operation (multiGPU) not work here, bc it uses tensors that are on CPU...
            return torch.Tensor(self.crf.decode(logits))
