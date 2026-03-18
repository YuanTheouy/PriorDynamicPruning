from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        
        # DEBUG PRINT (Only once per batch/step)
        if not hasattr(self, "debug_printed"):
            # print(f"DEBUG: CLP Call - Count: {self.count}")
            # print(f"DEBUG: CLP Input IDs Shape: {input_ids.shape}")
            self.debug_printed = True
            
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    # FIX: Logic error in original code?
                    # If count > 0, hash_key should be accumulating?
                    # The original logic: hash_key = sent[-self.count:]
                    # If count=1, takes last 1 token.
                    # If count=2, takes last 2 tokens.
                    # This implies Trie depth matches generation step?
                    # Trie stores: [Token1, Token2, Token3]
                    # Step 0: Input "Prefix", Key = "Prefix"[-3:] -> Expect Token1
                    # Step 1: Input "Prefix"+Token1, Key = "Token1" -> Expect Token2?
                    # Step 2: Input "Prefix"+Token1+Token2, Key = "Token1, Token2" -> Expect Token3?
                    
                    # BUT `hash_key` logic: `sent[-self.count:]`
                    # Step 1 (count=1): sent[-1:] -> [Token1]. Correct.
                    # Step 2 (count=2): sent[-2:] -> [Token1, Token2]. Correct.
                    
                    # Wait, is `self.count` incremented correctly?
                    # Yes, self.count += 1 at the end.
                    
                    hash_key=sent[-self.count:]
                
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    # if self.count < 3: # Reduce noise
                    #     warnings.warn(
                    #         f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                    #         f"This indicates the model generated an unexpected token. "
                    #     )
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue 
                
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        self.count += 1

        scores = scores + mask
        return scores