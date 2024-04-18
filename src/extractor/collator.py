from transformers.data import DataCollatorWithPadding
import torch

class DataCollatorForRelationClassification(DataCollatorWithPadding):

    def _convert_to_dict_of_samples(self, features):
        keys = features[0].keys()
        return {k:[data[k] for data in features] for k in keys}
         
    
    def __call__(self, features):

        if isinstance(features, list):
            features = self._convert_to_dict_of_samples(features)
        
        indexes = features.pop("indexes")

        padded_features = super().__call__(features)

        # list of indexes
        max_pad_len = max([len(sample_indexes) for sample_indexes in indexes])

        # unsqueeze bc of the gatther that we run on the model
        padded_indexes = torch.tensor([sample_indexes + [-1]*(max_pad_len-len(sample_indexes)) for sample_indexes in indexes], dtype=torch.int32).unsqueeze(-1)

        return padded_features | {"indexes": padded_indexes}


class DataCollatorForRelationAndNovelClassification(DataCollatorForRelationClassification):

    def __call__(self, features):

        if isinstance(features, list):
            features = self._convert_to_dict_of_samples(features)

        if "novel" in features:
            novel = features.pop("novel")
            # unsqueeze bc of the gatther that we run on the model
            padded_novel = {"novel":torch.tensor(novel, dtype=torch.int64)}
        else:
            padded_novel = {}

        padded_features = super().__call__(features)

        return padded_features | padded_novel


class DataCollatorForInference(DataCollatorForRelationAndNovelClassification):

    def __init__(self, inputs_keys, replicated_keys=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs_keys = inputs_keys 
        self.replicated_keys = replicated_keys
    
    def __call__(self, features):

        if isinstance(features, list):
            features = self._convert_to_dict_of_samples(features)

        model_features = {}
        metadata_features = {}
        
        for k in features.keys():
            if k in self.replicated_keys:
                model_features[k] = features[k]
                metadata_features[k] = features[k]
            elif k in self.inputs_keys:
                model_features[k] = features[k]
            else:
                metadata_features[k] = features[k]

        padded_features = super().__call__(model_features)

        return padded_features, metadata_features