# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]


class MaskedTransformerModel:
    """Wrapper for the custom MaskedTransformer (encoder-decoder).

    Loads a checkpoint saved by ``scripts/tmodel/train.py`` and exposes the
    same ``process_batch`` interface used by the RULER prediction pipeline.
    """

    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        import sys as _sys, os as _os

        # Make the tmodel package importable
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
        from tmodel import MaskedTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(name_or_path, map_location=self.device, weights_only=False)
        model_args = ckpt["args"]

        # Tokenizer
        tok_name = model_args.get("tokenizer", "gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Rebuild model from saved hyper-parameters
        self.max_len = model_args.get("max_len", 8192)
        self.model = MaskedTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=model_args["d_model"],
            num_heads=model_args["num_heads"],
            num_encoder_layers=model_args["num_layers"],
            num_decoder_layers=model_args["num_layers"],
            d_ff=model_args["d_ff"],
            pe_type=model_args["pe_type"],
            encoder_mask_type=model_args["encoder_mask"],
            decoder_mask_type=model_args["decoder_mask"],
            max_len=self.max_len,
            pad_token_id=self.tokenizer.pad_token_id or 0,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        self.stop = generation_kwargs.pop("stop", [])
        self.max_new_tokens = generation_kwargs.pop("max_new_tokens", 64)
        self.temperature = generation_kwargs.get("temperature", 0.0)
        self.top_k = generation_kwargs.get("top_k", 0)

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self.max_len,
            )
            input_ids = inputs.input_ids.to(self.device)

            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                bos_token_id=self.tokenizer.bos_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=self.temperature,
                top_k=self.top_k,
            )

            # Decode (skip the leading BOS token the decoder started with)
            text = self.tokenizer.decode(output_ids[0, 1:], skip_special_tokens=True)

            for s in self.stop:
                text = text.split(s)[0]

            results.append({"text": [text]})
        return results
