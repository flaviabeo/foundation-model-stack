from typing import Optional, Tuple
from typing_extensions import Unpack

from fms.modules.attention import SDPAAttentionKwargs
import torch
import torch.nn as nn
from transformers import PretrainedConfig
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
        if kwargs.get("mask", None) is None:
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
            ]
            for param in hf_only_params:
                params.pop(param, None)

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
        **model_kwargs,
    ) -> dict:
        """
        Overriding _prepare_inputs_for_generation to include position_ids requirements for Mistral batch processing
        """
        position_ids = model_kwargs.pop("position_ids", None)

        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # Add more cached rope freqs if over cached number
        if position_ids is not None:
            max_expected_len = input_ids.shape[1] + torch.max(position_ids).item()
            if (
                max_expected_len
                > self.decoder.model.rot_emb.rope_scaling.orig_max_seq_len
            ):
                self.decoder.model.rot_emb.compute_freqs_cis(
                    input_ids.device, max_expected_len
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            **model_kwargs,
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
        return out

    @classmethod
    def from_fms_model(
        cls, model: Mistral, **hf_config_kwargs
    ) -> "HFAdaptedMistralForCausalLM":
        """
        This method is used to convert an FMS model to an HF model. It is used in the
        `to_hf_api` method.

        Parameters
        ----------
        model: Mistral
            the FMS model to convert
        hf_config_kwargs
            additional configuration parameters to pass to the HF config

        Returns
        -------
        HFAdaptedMistralForCausalLM
            the HF model
        """
        config = HFAdaptedMistralConfig.from_fms_config(
            model.get_config(), **hf_config_kwargs
        )
        return cls._hf_model_from_fms(model, config)
