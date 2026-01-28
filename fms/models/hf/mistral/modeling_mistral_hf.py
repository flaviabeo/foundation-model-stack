from typing import Optional, Tuple
from typing_extensions import Unpack

from fms.modules.attention import SDPAAttentionKwargs
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.mistral.configuration_mistral_hf import HFAdaptedMistralConfig
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.mistral import Mistral, MistralHeadless, MistralConfig


class HFAdaptedMistralDecoder(HFDecoder):
    """Adapter for the Mistral decoder"""

    def __init__(self, model: MistralHeadless, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        *args,
        **kwargs: Unpack[SDPAAttentionKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if kwargs.get("mask", None) is None and attention_mask is not None:
            kwargs["mask"] = attention_mask


        output = self.model(
            x_in=input_ids,
            position_ids=position_ids,
            past_key_value_states=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        present_key_values = None
        if isinstance(output, tuple):
            output, present_key_values = output
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output, past_key_values=present_key_values
        )

class HFAdaptedMistralHeadless(HFDecoderModelArchitecture):
    """This is the Adapter for the base Mistral architecture"""

    # attributes required by HF
    config_class = HFAdaptedMistralConfig
    base_model_prefix = "hf_adapted_mistral"

    _tied_weights_keys = ["decoder.model.embedding.weight", "embedding.weight"]
    _keys_to_ignore_on_save = ["embedding.weight"]

    def __init__(
        self,
        config: PretrainedConfig,
        decoder: Optional[nn.Module] = None,
        embedding: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        # in the case we have not yet received the encoder/decoder/embedding, initialize it here
        if decoder is None or embedding is None:
            params = config.to_dict()
            params["pad_id"] = params.pop("pad_token_id")

            # Remove HF-specific parameters that are not part of MistralConfig
            hf_only_params = [
                "use_cache",
                "is_decoder",
                "add_cross_attention",
                "architectures",
                "transformers_version",
                "_name_or_path",
                "torch_dtype",
                "auto_map",
                "model_type",
                "return_dict",
                "output_hidden_states",
                "output_attentions",
                "use_return_dict",
                "tie_word_embeddings",
                "torchscript",
                "dtype", "pruned_heads",
                "chunk_size_feed_forward",
                "is_encoder_decoder",
                "cross_attention_hidden_size",
                "tie_encoder_decoder", 
                "finetuning_task", 
                "id2label",
                "label2id",
                "task_specific_params",
                "problem_type",
                "tokenizer_class",
                # Generation-related
                "max_length",
                "min_length", 
                "do_sample",
                "prefix",
                "early_stopping",
                "num_beams",
                "num_beam_groups",
                "diversity_penalty",
                "temperature",
                "top_k",
                "top_p",
                "typical_p",
                "repetition_penalty",
                "length_penalty",
                "no_repeat_ngram_size",
                "encoder_no_repeat_ngram_size",
                "bad_words_ids",
                "num_return_sequences",
                "output_scores",
                "return_dict_in_generate",
                "forced_bos_token_id",
                "forced_eos_token_id",
                "remove_invalid_values",
                "exponential_decay_length_penalty",
                "suppress_tokens",
                "begin_suppress_tokens",

                # Tokenizer-related
                "bos_token_id",
                "eos_token_id", 
                "pad_token_id",  # Note: Mistral converts this to pad_id
                "sep_token_id",
                "decoder_start_token_id",

                # Training-related
                "gradient_checkpointing",
                "use_cache",

                # Quantization-related  
                "quantization_config",
                "pretraining_tp",

                # Other metadata
                "name_or_path",
                "_commit_hash",
                "_attn_implementation",
                "attn_implementation",
                "tf_legacy_loss",
                "use_bfloat16"

            ]
            for param in hf_only_params:
                if param in params:
                    del params[param]

            fms_config = MistralConfig(**params)
            model = MistralHeadless(fms_config)
            decoder = model if decoder is None else decoder
            embedding = model.embedding if embedding is None else embedding

        # these are now huggingface compatible
        decoder = HFAdaptedMistralDecoder(decoder, config)
        super().__init__(decoder, embedding, config, *args, **kwargs)

    def _prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        """
        Overriding _prepare_inputs_for_generation to include position_ids requirements for Mistral batch processing
        """
        position_ids = kwargs.pop("position_ids", None)

        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs

        # Check if cache has content
        has_cache = False
        if past_key_values is not None:
            if isinstance(past_key_values, DynamicCache):
                # DynamicCache has content if get_seq_length returns > 0
                has_cache = past_key_values.get_seq_length() > 0
            else:
                # Tuple format - has content if not None
                has_cache = True

        if has_cache:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)

        # Add more cached rope freqs if over cached number
        max_expected_len = input_ids.shape[1] + torch.max(position_ids)
        if max_expected_len > self.decoder.model.rot_emb.rope_scaling.orig_max_seq_len:
            self.decoder.model.rot_emb.compute_freqs_cis(
                input_ids.device, max_expected_len
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            **kwargs,
        }


class HFAdaptedMistralForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedMistralHeadless):
    """
    This is the Adapter for Mistral for Causal LM. It is a composition of the
    HFAdaptedMistralHeadless and the LMHeadModelLMHeadMixin.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["embedding.weight", "lm_head.weight"]

    def __init__(self, config: HFAdaptedMistralConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    @classmethod
    def _hf_model_from_fms(
        cls, model: Mistral, config: HFAdaptedMistralConfig
    ) -> "HFAdaptedMistralForCausalLM":
        out = cls(
            config=config,
            decoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.head,
        )
        # Ensure RoPE frequencies are recomputed on the correct device
        # This is necessary because the decoder was already initialized in the FMS model
        out.decoder.model.post_init()
        return out

