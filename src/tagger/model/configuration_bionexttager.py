
from transformers import PretrainedConfig, AutoConfig
from typing import List


class BioNExtTaggerConfig(PretrainedConfig):
    model_type = "crf-tagger"

    def __init__(
        self,
        augmentation = "unk",
        context_size = 64,
        percentage_tags = 0.2,
        p_augmentation = 0.5,
        crf_reduction = "mean",
        freeze = False,
        version="0.1.2",
        **kwargs,
    ):
        self.version = version
        self.augmentation = augmentation
        self.context_size = context_size
        self.percentage_tags = percentage_tags
        self.p_augmentation = p_augmentation
        self.crf_reduction = crf_reduction
        self.freeze=freeze
        super().__init__(**kwargs)
        

        