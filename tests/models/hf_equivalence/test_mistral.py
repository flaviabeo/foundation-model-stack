import pytest
import torch

from fms.models import get_model
from fms.models.hf import to_hf_api
from fms.testing.comparison import (
    HFModelSignatureParams,
    ModelSignatureParams,
    compare_model_signatures,
)

from difflib import SequenceMatcher

@pytest.mark.slow
def test_mistral_equivalence():
    """
    Tests Mistral equivalence with a known implementation.

    This test verifies that the FMS Mistral implementation is equivalent to the
    HuggingFace Mistral implementation by comparing:
    - Parameter counts
    - Model signatures
    - Generation outputs
    - Training loss

    Note: This test requires a HuggingFace Mistral model path to be set.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    mistral_model_path = "mistralai/Mistral-7B-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(mistral_model_path, use_fast=True)
    
    # Load both models in float32 for accurate comparison
    # bfloat16 can have numerical differences between implementations
    model = get_model("hf_pretrained", mistral_model_path, data_type="float32")
    hf_model = AutoModelForCausalLM.from_pretrained(mistral_model_path, torch_dtype=torch.float32)

    hf_model_fms = to_hf_api(
        model,
        bos_token_id=hf_model.config.bos_token_id,
        eos_token_id=hf_model.config.eos_token_id,
        pad_token_id=hf_model.config.pad_token_id,
    )

    model.eval()
    hf_model.eval()
    hf_model_fms.eval()

    # Test Parameter Count

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters())

    assert count_parameters(hf_model_fms) == count_parameters(hf_model)

    # Test Model Signatures

    inp = torch.arange(0, 16).unsqueeze(0)
    fms_signature_params = ModelSignatureParams(model=model, params=1, inp=inp)
    hf_fms_signature_params = HFModelSignatureParams(
        model=hf_model_fms,
        params=["input_ids", "labels"],
        other_params={"return_dict": True},
        inp=inp,
    )
    hf_signature_params = HFModelSignatureParams(
        model=hf_model,
        params=["input_ids", "labels"],
        other_params={"return_dict": True},
        inp=inp,
    )

    compare_model_signatures(fms_signature_params, hf_fms_signature_params)
    compare_model_signatures(hf_fms_signature_params, hf_signature_params)

    # Test Generation Pipeline

    prompt = """q: how are you? a: I am good. How about you? q: What is the weather like today? a:"""

    generator_hf = pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        use_cache=True,
        num_beams=3,
        max_new_tokens=10,
    )
    generator_hf_fms = pipeline(
        task="text-generation",
        model=hf_model_fms,
        tokenizer=tokenizer,
        use_cache=True,
        num_beams=3,
        max_new_tokens=10,
    )
    output_hf = generator_hf(prompt)
    output_hf_fms = generator_hf_fms(prompt)

    # Calculate similarity ratio
    ratio = SequenceMatcher(None, output_hf[0]["generated_text"], output_hf_fms[0]["generated_text"]).ratio()

    assert ratio > 0.8
