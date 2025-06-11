from accelerate import Accelerator
import yaml
from datetime import datetime
from transformers import StoppingCriteria, EosTokenCriteria
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    set_seed,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GenerationConfig,
    StoppingCriteriaList
)
import tyro
from src.data.pref_dataset import *
from src.scripts.config import EvaluateConfig
from src.scripts.utils import *
import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.autograd.grad_mode.set_grad_enabled(False)

def forward(sequences, model, tokenizer):
    attention_mask = sequences != tokenizer.pad_token_id
    input_ids = torch.masked_fill(sequences, ~attention_mask, 0)
    position_ids = torch.cumsum(attention_mask, dim=-1) - 1
    position_ids.masked_fill_(~attention_mask, 0)
    policy_logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    ).logits.to(torch.float32)
    return policy_logits

if __name__=="__main__":
    config = tyro.cli(EvaluateConfig)
    set_seed(config.seed)
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parent_path = os.path.dirname(config.model_path)
    train_config_path = os.path.join(parent_path, 'config.yaml')
    try:
        with open(train_config_path, 'r') as f:
            training_config = yaml.safe_load(f)
    except:
        training_config = None
    # load_in_8bit=True

    reward_model = AutoModelForSequenceClassification.from_pretrained(config.judge_path, torch_dtype=torch.bfloat16, device_map='auto')
    rm_tokenizer = AutoTokenizer.from_pretrained(config.judge_path)
    if rm_tokenizer.pad_token_id is None:
        rm_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    policy_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": 'auto',
        "use_cache": True
    }

    policy = AutoModelForCausalLM.from_pretrained(config.model_path, **policy_kwargs)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.ref_model_path, **policy_kwargs)
    eval_dataset = globals()[f'get_{config.dataset_name}']('test')
    eval_dataset = eval_dataset.shuffle(seed=config.seed)
    eval_dataset = eval_dataset.select(range(config.num_samples))
    eval_dataset = RewardDataset(eval_dataset, tokenizer)

    world_size = accelerator.num_processes
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, collate_fn=eval_dataset.padding_collate_fn, pin_memory=True, num_workers=4, shuffle=False)
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=config.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=2,  
    )

    policy, ref_policy, reward_model, eval_dataloader = accelerator.prepare(policy, ref_policy, reward_model, eval_dataloader)
    all_kls, all_scores, all_ref_scores = [], [], []
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    all_policy_logps = []
    all_ref_policy_logps = []
    i = 0

    for batch in tqdm(eval_dataloader):
        context_length = batch['prompt_input_ids'].size(1)
        policy_output = get_batch_samples(batch, policy, tokenizer, return_tensors=True, generation_config=generation_config)
        policy_logits = forward(policy_output, policy, tokenizer)
        policy_logps = F.log_softmax(policy_logits, dim=-1)[:, context_length-1:-1, :]
        ref_policy_logits = forward(policy_output, ref_policy, tokenizer)
        ref_policy_logps = F.log_softmax(ref_policy_logits, dim=-1)[:, context_length-1:-1, :]

        ref_policy_logps = torch.gather(ref_policy_logps, dim=2, index=policy_output[:, context_length:].unsqueeze(2)).squeeze(2)
        policy_logps = torch.gather(policy_logps, dim=2, index=policy_output[:, context_length:].unsqueeze(2)).squeeze(2)
        loss_mask = (policy_output[:, context_length:] != tokenizer.pad_token_id)

        policy_logps = (policy_logps * loss_mask).sum(1)
        ref_policy_logps = (ref_policy_logps * loss_mask).sum(1)

        estimated_kl = (policy_logps - ref_policy_logps) + (torch.exp(ref_policy_logps - policy_logps) - 1)
        estimated_kl = estimated_kl.float().detach().cpu().numpy().tolist()
        
        policy_output_decoded = tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        policy_output_decoded = [ele.strip() + ' ' + rm_tokenizer.eos_token for ele in policy_output_decoded]

        
        batch_prompt = batch['prompt_text']

        reward_inputs = rm_tokenizer(
            policy_output_decoded,
            padding=True, 
            truncation=True,
            max_length=640,
            return_tensors="pt",
        ).to(reward_model.device)

        ref_output_decoded = [ele.replace(tokenizer.eos_token, "").strip() + ' ' + rm_tokenizer.eos_token for ele in batch['chosen_combined_text']]

        if i == 0:
            print(policy_output_decoded[0])
            print("--"*80)
            print(ref_output_decoded[0])
            print("--"*80)
            print(policy_output_decoded[2])
            print("--"*80)
            print(ref_output_decoded[1])
            i += 1
        ref_reward_inputs = rm_tokenizer(
            ref_output_decoded,
            padding=True, 
            truncation=True,
            max_length=640,
            return_tensors="pt",
        ).to(reward_model.device)

        reward_logits = reward_model(reward_inputs['input_ids'], attention_mask=reward_inputs['attention_mask']).logits.squeeze(-1)
        reward_scores = reward_logits.float().detach().cpu().numpy().tolist()
        ref_reward_logits = reward_model(ref_reward_inputs['input_ids'], attention_mask=ref_reward_inputs['attention_mask']).logits.squeeze(-1)
        ref_reward_scores = ref_reward_logits.float().detach().cpu().numpy().tolist()
        all_ref_scores.extend(ref_reward_scores)
        all_scores.extend(reward_scores)
        all_policy_logps.extend(policy_logps.float().cpu().numpy().tolist())
        all_ref_policy_logps.extend(ref_policy_logps.float().cpu().numpy().tolist())
        all_kls.extend(estimated_kl)
    
    all_scores = np.array(all_scores)
    ref_scores = np.array(all_ref_scores).reshape(-1, 1)
    all_scores = all_scores.reshape(-1, generation_config.num_return_sequences)
    wins = (all_scores > ref_scores).mean(axis=1)
    wins = wins.tolist()
    
    results = {
        'date': str(datetime.now()),
        'total': len(eval_dataset),
        "dataset_name": config.dataset_name,
        'judge' : config.judge_path,
        'seed': config.seed,
        'name': config.key_name,
        'win_rate': sum(wins) / len(wins),
        'kl_div': sum(all_kls) / len(all_kls),
        'scores': all_scores.flatten().mean(),
        'ref_wins': 1 - sum(wins) / len(wins),
        'ref_scores': ref_scores.flatten().mean(),
        'ref_logprobs': sum(all_ref_policy_logps) / len(all_ref_policy_logps),
        "policy_logprobs": sum(all_policy_logps) / len(all_policy_logps),
        'config': training_config
    }

    with open(config.result_path, 'a+') as f:
        json.dump(results, f)
        f.write('\n')

    