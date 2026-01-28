import pytest
import torch
import itertools

from typing import List, Optional

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from difflib import SequenceMatcher

from fms.models.hf.mistral.configuration_mistral_hf import HFAdaptedMistralConfig
from fms.models.hf.mistral.modeling_mistral_hf import HFAdaptedMistralForCausalLM
from fms.models.mistral import Mistral
from fms.models.hf.mistral import convert_to_hf
from fms.testing._internal.hf.model_test_suite import (
    HFAutoModelTestSuite,
    HFConfigFixtureMixin,
    HFConfigTestSuite,
    HFModelCompileTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelFixtureMixin,
    HFModelGenerationTestSuite,
)
from fms.testing._internal.model_test_suite import ModelFixtureMixin

from ..test_mistral import MistralFixtures


class MistralHFFixtures(ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(self, model: Mistral, fms_hf_config: PretrainedConfig, **kwargs):
        return HFAdaptedMistralForCausalLM.from_fms_model(
            model, **fms_hf_config.to_dict()
        )

    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self, tokenizer: PreTrainedTokenizer, model: Mistral, **kwargs
    ) -> PretrainedConfig:
        bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        return HFAdaptedMistralConfig.from_fms_config(
            model.get_config(),
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=bos_token_id,
        )

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(
        self, fms_hf_model: HFAdaptedMistralForCausalLM
    ) -> PreTrainedModel:
        return convert_to_hf(fms_hf_model)


class TestMistralHF(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
    HFAutoModelTestSuite,
    MistralFixtures,
    MistralHFFixtures,
):
    """
    Mistral FMS Huggingface Tests for:

    - FMS Huggingface configuration tests
    - model equivalency tests
    - model generation tests
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]

    def _predict_text(self, model, tokenizer, texts, use_cache, num_beams):
        encoding = tokenizer(texts, padding=True, return_tensors="pt")

        # Fix for newer versions of transformers
        use_cache_kwarg = {}
        if use_cache is not None:
            use_cache_kwarg["use_cache"] = use_cache

        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                num_beams=num_beams,
                max_new_tokens=10,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=model.config.pad_token_id,
                **use_cache_kwarg,
            )
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_texts
    
    text_options = [
        ["hello how are you?"],
        ["hello how are you?", "a: this is a test. b: this is another test. a:"],
    ]
    use_cache_options = [True, False, None]
    num_beams_options = [1, 3]
    generate_equivalence_args = list(
        itertools.product(text_options, use_cache_options, num_beams_options)
    )

    @pytest.mark.parametrize("texts,use_cache,num_beams", generate_equivalence_args)
    def test_hf_generate_equivalence(
        self,
        texts: List[str],
        use_cache: Optional[bool],
        num_beams: int,
        fms_hf_model: PreTrainedModel,
        oss_hf_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """test that an hf model created from fms and an hf model loaded from hf checkpoint produce the same output if
        they have the same weights and configs
        """
        print(texts)
        output_fms = self._predict_text(
            fms_hf_model, tokenizer, texts, use_cache, num_beams
        )
        output_hf = self._predict_text(
            oss_hf_model, tokenizer, texts, use_cache, num_beams
        )

        # Calculate similarity ratio
        ratio = SequenceMatcher(None, output_hf[0], output_hf[0]).ratio()

        assert ratio > 0.9

    hf_batch_generate_args = list(
        itertools.product(use_cache_options, num_beams_options)
    )

    @pytest.mark.parametrize("use_cache,num_beams", hf_batch_generate_args)
    def test_hf_batch_generate(
        self,
        use_cache,
        num_beams,
        fms_hf_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Tests that the output of a given prompt done alone and with batch generation is the same"""
        text_1 = "hello how are you?"
        text_2 = "a: this is a test. b: this is another test. a:"
        text_batch = [text_1, text_2]

        output_batch = self._predict_text(
            fms_hf_model, tokenizer, text_batch, use_cache, num_beams
        )

        text1 = [text_1]
        output_text1 = self._predict_text(
            fms_hf_model, tokenizer, text1, use_cache, num_beams
        )[0]

        text2 = [text_2]
        output_text2 = self._predict_text(
            fms_hf_model, tokenizer, text2, use_cache, num_beams
        )[0]

        ratio1 = SequenceMatcher(None, output_batch[0], output_text1).ratio()

        assert ratio1 > 0.9, f"text 1 incorrect - \n{output_batch[0]}\n{output_text1}"

        ratio2 = SequenceMatcher(None, output_batch[1], output_text2).ratio()

        assert ratio2 > 0.9, f"text 2 incorrect - \n{output_batch[1]}\n{output_text2}"
 

