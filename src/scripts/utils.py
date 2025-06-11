import torch
from transformers import GenerationConfig
from accelerate.state import AcceleratorState
import deepspeed
import numpy as np
import gc
from contextlib import contextmanager
import itertools
import os
from typing import Optional
import json
from typing import Dict
from typing import Dict, Union, Type, List, TextIO
from transformers import StoppingCriteria


class StreamingJSONWriter:
    """Writes JSON arrays to a file in a streaming fashion."""
    def __init__(self, file: TextIO):
        self.file = file
        self.is_first = True
        # self.file.write('[\n')
    
    def write_item(self, item: Dict):
        """Write a single item to the JSON array."""
        # if not self.is_first:
        #     self.file.write(',\n')
        json.dump(item, self.file)
        self.file.write('\n')
        # self.is_first = False
        # Flush after each write to ensure immediate disk writing
        self.file.flush()
    
    def close(self):
        """Close the JSON array and the file."""
        # self.file.write('\n]')
        self.file.flush()


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}


def delete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]


def delete_dicts(*dicts: Dict):
    """Delete all items inside the given dictionaries."""
    for d in dicts:
        for k in list(d.keys()):
            del d[k]
        
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


@torch.no_grad()
def get_batch_samples(batch, policy, tokenizer, return_tensors=False, **generation_kwargs):
    policy_output = policy.generate(
        batch['prompt_input_ids'],
        attention_mask=batch['prompt_attention_mask'],
        **generation_kwargs
    )
    if return_tensors:
        return policy_output
    else:
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        return policy_output_decoded

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

def save(policy, example_counter, accelerator, tokenizer, config, metrics: Optional[Dict] = {}):
    output_dir = os.path.join(config.output_dir, f'step-{example_counter}')
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        accelerator.print(f"Saving tokenizer...")
        tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            metrics['counter'] = example_counter
            json.dump(metrics, f)
    
    accelerator.wait_for_everyone()
    accelerator.print(f"Saving model...")
    
    state_dict = accelerator.get_state_dict(policy)
    unwrapped_model = accelerator.unwrap_model(policy)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=False
    )
    accelerator.wait_for_everyone()
    

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer

def concatneted_inputs(padded_batch, tokenizer):
    max_length = max(padded_batch['chosen_combined_input_ids'].shape[1], padded_batch['rejected_combined_input_ids'].shape[1])
    concatenated_batch = {}

    for k in padded_batch:
        if k.startswith('chosen') and isinstance(padded_batch[k], torch.Tensor):
            if 'labels' in k:
                pad_value = -100 
            elif 'input_ids' in k:
                pad_value = tokenizer.pad_token_id
            elif 'attention_mask' in k:
                pad_value = 0

            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(padded_batch[k], max_length, pad_value=pad_value)

    for k in padded_batch:
        if k.startswith('rejected') and isinstance(padded_batch[k], torch.Tensor):
            if 'labels' in k:
                pad_value = -100 
            elif 'input_ids' in k:
                pad_value = tokenizer.pad_token_id
            elif 'attention_mask' in k:
                pad_value = 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(padded_batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

def prepare_deepspeed(model, config):
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = config.batch_size // torch.cuda.device_count()

    eval_ds_config = {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
        "bf16": {"enabled": True},
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }

    # eval_ds_config["zero_optimization"] = {
    #     "stage": 3,
    #     "stage3_param_persistence_threshold": 1e4,
    # }
    model, *_ = deepspeed.initialize(model=model, config=eval_ds_config)
    model.eval()
    return model


def get_completion_mask(completion_ids, tokenizer, device):
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    return completion_mask


def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]

def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []

def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)

@contextmanager
def unwrap_model_for_generation(model, accelerator, gather_deepspeed3_params: bool = True):
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        unwrapped_model = accelerator.unwrap_model(model)
        yield unwrapped_model


def selective_log_softmax(logits, index):
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    return per_token_logps

def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def truncate_right(
    input_ids: torch.Tensor, stop_token_id: int, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates the input tensor from the right side after the first occurrence of the stop token.

    Args:
        input_ids (`torch.Tensor`):
            The tensor containing the responses to be truncated
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses

    Returns:
        tuple:
            - `output_ids` (`torch.Tensor`):
                The truncated responses tensor with pad tokens filled after the stop token
            - `mask` (`torch.Tensor`):
                The mask tensor to indicate the padding tokens
    """
    trunc_idxs = first_true_indices(input_ids == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(input_ids.size()) - 1) + [input_ids.shape[1]]
    idxs = torch.arange(input_ids.shape[1], device=input_ids.device).view(*new_size)
    output_ids = torch.masked_fill(input_ids, idxs > trunc_idxs, pad_token_id)
    mask = torch.masked_fill(torch.ones_like(input_ids), idxs > trunc_idxs, 0)
    return output_ids, mask

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="right")
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss

def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits