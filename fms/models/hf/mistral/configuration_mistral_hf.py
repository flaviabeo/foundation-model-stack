from typing import Optional

from transformers import PretrainedConfig

from fms.models.mistral import MistralConfig


class HFAdaptedMistralConfig(PretrainedConfig):
    model_type = "hf_adapted_mistral"
    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "emb_dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
        "num_key_value_heads": "kvheads",
        "intermediate_size": "hidden_grow_factor",
        "rms_norm_eps": "norm_eps",
        "max_position_embeddings": "max_expected_seq_len",
        "rope_theta": "rope_base",
        "attention_dropout": "p_dropout",
    }

    def __init__(
        self,
        src_vocab_size: Optional[int] = 32768,
        emb_dim: Optional[int] = 4096,
        nheads: int = 32,
        nlayers: int = 32,
        hidden_grow_factor: float = 14336 / 4096,
        multiple_of: int = 256,
        tie_heads: bool = False,
        p_dropout: float = 0.0,
        activation_fn: str = "swish",
        head_dim: int = 128,
        max_expected_seq_len: int = 32768,
        kvheads: int = 8,
        norm_eps: float = 1e-5,
        sliding_window: int = 4000,
        rope_base: float = 1000000.0,
        rope_scaling: dict = {},
        use_cache: bool = True,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        pad_token_id: int = 0,
        is_decoder: bool = True,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.nlayers = nlayers
        self.hidden_grow_factor = hidden_grow_factor
        self.multiple_of = multiple_of
        self.tie_heads = tie_heads
        self.p_dropout = p_dropout
        self.activation_fn = activation_fn
        self.head_dim = head_dim
        self.max_expected_seq_len = max_expected_seq_len
        self.kvheads = kvheads
        self.norm_eps = norm_eps
        self.sliding_window = sliding_window
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        self.use_cache = use_cache
        # Extract tie_word_embeddings before passing kwargs to parent
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", tie_heads)
        
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_fms_config(cls, config: MistralConfig, **hf_kwargs):
        config_dict = config.as_dict()
        config_dict["pad_token_id"] = config_dict.pop("pad_id")
        return cls.from_dict(config_dict, **hf_kwargs)
