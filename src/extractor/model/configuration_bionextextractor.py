
from transformers import PretrainedConfig, AutoConfig
from typing import List


class BioNExtExtractorConfig(PretrainedConfig):
    model_type = "relation-novelty-extractor"

    def __init__(
        self,
        arch_type = "mha",
        index_type = "both",
        novel = True,
        tokenizer_special_tokens = ['[s1]','[e1]', '[s2]','[e2]' ],
        update_vocab = None,
        version="0.1.1",
        **kwargs,
    ):
        self.version = version
        self.arch_type = arch_type
        self.index_type = index_type
        self.novel = novel
        self.tokenizer_special_tokens = tokenizer_special_tokens
        self.update_vocab = update_vocab
        super().__init__(**kwargs)
        

        