import accelerate
from accelerate import Accelerator,InitProcessGroupKwargs
import argparse
import datasets
import datetime
from huggingface_hub import login
import json
import logging
import numpy as np
import os
from termcolor import colored
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    GPT2Tokenizer,
    OPTForCausalLM
)
import torch
import torch.nn.functional as F  # 新添加的
from tqdm import tqdm


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, torch.finfo(logits.dtype).min)
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return (2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value)


def filter_logits_top_p(logits, top_p, negative_multiplier=False):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    if negative_multiplier:
        filtered_logits = logits.masked_fill(valid_indices == 0, 1000)
    else:
        filtered_logits = logits.masked_fill(valid_indices == 0, -1000)
    return filtered_logits

def get_context_token_ids(context_input_ids):
    """获取context中的unique token ids"""
    return torch.unique(context_input_ids)

def create_boost_mask(logits, context_tokens, delta):
    """创建boost mask用于增强context tokens的权重"""
    device = logits.device
    if not torch.is_tensor(delta):
        delta = torch.tensor(delta, device=device)
    boost_mask = torch.zeros_like(logits)
    for token in context_tokens:
        boost_mask[..., token] = delta
    return boost_mask


def compute_jsd(logits_with_context, logits_without_context):
    """计算Jensen-Shannon散度"""
    p = F.softmax(logits_with_context, dim=-1)
    q = F.softmax(logits_without_context, dim=-1)
    m = 0.5 * (p + q)
    
    jsd = 0.5 * (F.kl_div(m.log(), p, reduction='batchmean') + 
                 F.kl_div(m.log(), q, reduction='batchmean'))
    return jsd

def compute_entropy(logits):
    """计算条件熵"""
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy

def get_adaptive_delta(logits_with_context, logits_without_context, args):
    """根据JSD动态确定boost delta值"""
    jsd = compute_jsd(logits_with_context, logits_without_context)
    
    # 将JSD映射到delta范围
    args.min_delta = 1.0
    args.max_delta = 10.0
    delta = args.min_delta + (args.max_delta - args.min_delta) * jsd
    
    return delta

def get_adaptive_temperature(logits, args):
    """根据条件熵动态确定温度"""
    entropy = compute_entropy(logits)
    
    # 将熵映射到温度范围
    temp = args.min_temp + args.base_temp * entropy
    
    # 确保返回标量值
    if torch.is_tensor(temp):
        temp = temp.item()
    
    return temp

def decode(args, batch_input_ids, dec_depth, model, tokenizer):
    batch_size = args.per_device_eval_batch_size
    assert batch_input_ids.size(1) == args.context_size
    assert args.decode_truncate_len >= 0
    assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)

    # 获取context input和question input
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        question_input_ids = batch_input_ids[:, args.context_size:].clone()
        context_tokens = get_context_token_ids(unit_context_input_ids[0])
    else:
        raise ValueError("context cannot be none")
    
    history_decode_ids = None
    past_key_values = None

    # 初始化统计信息字典
    stats = {
        'jsd_values': [],
        'entropy_values': [],
        'delta_values': [],
        'temperature_values': []
    }

    if args.model_category == 'seq2seq':
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            batch_input_ids[:, :args.context_size].clone(), dict(), None
        )
        history_decode_ids = model._prepare_decoder_input_ids_for_generation(
            batch_input_ids.size(0),
            model_kwargs=model_kwargs,
            device=batch_input_ids.device,
        )
    else:
        model_kwargs = None

    for _i in range(dec_depth):
        if args.model_category == 'causal':
            if 'llama' in args.model_name_or_path.lower():
                if past_key_values is not None:
                    model_inputs_with_context = {
                        "input_ids": unit_context_input_ids[:, -1:],
                        "attention_mask": torch.ones_like(unit_context_input_ids),
                        "past_key_values": past_key_values,
                        "use_cache": True
                    }
                else:
                    model_inputs_with_context = {
                        "input_ids": unit_context_input_ids,
                        "attention_mask": torch.ones_like(unit_context_input_ids),
                        "use_cache": True
                    }
                
                # 修改这部分以保持维度一致
                curr_token = unit_context_input_ids[:, -1:] if past_key_values is not None else unit_context_input_ids
                model_inputs_without_context = {
                    "input_ids": curr_token,
                    "attention_mask": torch.ones_like(curr_token),
                    "use_cache": True
                }
            else:
                model_inputs_with_context = model.prepare_inputs_for_generation(
                    unit_context_input_ids, 
                    past_key_values=past_key_values
                )
                curr_token = unit_context_input_ids[:, -1:] if past_key_values is not None else unit_context_input_ids
                model_inputs_without_context = model.prepare_inputs_for_generation(
                    curr_token,
                    past_key_values=None
                )
        elif args.model_category == 'seq2seq':
            model_inputs_with_context = model.prepare_inputs_for_generation(
                history_decode_ids, 
                **model_kwargs
            )
            model_inputs_without_context = model.prepare_inputs_for_generation(
                history_decode_ids,
                **model_kwargs 
            )
        else:
            raise ValueError("model category not supported")

        outputs_with_context = model(**model_inputs_with_context, output_hidden_states=False)
        outputs_without_context = model(**model_inputs_without_context, output_hidden_states=False)

        logits_with_context = outputs_with_context.logits[:, -1:, :].clone().contiguous()
        logits_without_context = outputs_without_context.logits[:, -1:, :].clone().contiguous()

        # 获取自适应参数
        if args.adaptive_mode in ["delta_only", "both"]:
            delta = get_adaptive_delta(logits_with_context, logits_without_context, args)
            if torch.is_tensor(delta):
                delta = delta.item()
        else:
            delta = args.fixed_delta

        if args.adaptive_mode in ["temp_only", "both"]:
            temperature = get_adaptive_temperature(logits_with_context, args)
        else:
            temperature = args.fixed_temp

        # 记录统计值
        if args.stats_logging:
            stats['jsd_values'].append(compute_jsd(logits_with_context, logits_without_context).item())
            stats['entropy_values'].append(compute_entropy(logits_with_context).item())
            stats['delta_values'].append(float(delta))  # 确保是Python float
            stats['temperature_values'].append(float(temperature))  # 确保是Python float

        # 将标量转换为tensor用于计算
        delta_tensor = torch.tensor(delta, device=logits_with_context.device)
        temperature_tensor = torch.tensor(temperature, device=logits_with_context.device)

        # 创建并应用boost mask和temperature
        boost_mask = create_boost_mask(logits_with_context, context_tokens, delta_tensor)
        enhanced_logits = logits_with_context + boost_mask
        enhanced_logits = enhanced_logits / temperature_tensor

        # 应用过滤
        if args.assigned_weight >= 0:
            score = filter_logits_top_p(enhanced_logits, top_p=args.filter_top_p)
        else:
            score = filter_logits_top_p(enhanced_logits, top_p=args.filter_top_p_prior, negative_multiplier=True)

        # 采样
        projected_logits = logits_sampling_projection(score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
        
        # 计算token IDs
        simplex = torch.nn.functional.softmax(projected_logits, dim=-1)
        real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)

        # 更新历史
        if args.model_category == 'causal':
            unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
            question_input_ids = torch.cat((question_input_ids, real_token_ids_list), dim=1)

        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

        if args.model_category == 'causal':
            past_key_values = outputs_with_context.past_key_values
        elif args.model_category == 'seq2seq':
            model_kwargs["past_key_values"] = outputs_with_context.past_key_values

        # 检查是否生成结束
        if real_token_ids_list[0][-1] == model.generation_config.eos_token_id:
            break

    # 计算平均统计值
    avg_stats = {
        'avg_jsd': sum(stats['jsd_values']) / len(stats['jsd_values']),
        'avg_entropy': sum(stats['entropy_values']) / len(stats['entropy_values']),
        'avg_delta': sum(stats['delta_values']) / len(stats['delta_values']),
        'avg_temperature': sum(stats['temperature_values']) / len(stats['temperature_values'])
    }

    # 处理输出
    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
    else:
        init_context_input_ids = None
        context_sequences = None

    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'), skip_special_tokens=True)
    logger.info(f"context: {context_sequences}")
    logger.info(f"sampled: {colored(str(sampled_sequences), 'red')}")
    logger.info(f"Statistics: {avg_stats}")

    return history_decode_ids, init_context_input_ids, avg_stats, sampled_sequences, context_sequences, stats

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")

    # 基础模型参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    
    # 文件和训练模式参数
    parser.add_argument(
        "--init_blank_language_model", 
        action="store_true",
        help="Whether or not to use a completely blank LM."
    )
    parser.add_argument(
        "--file_mode", 
        type=str, 
        default="", 
        help="File mode configuration"
    )
    parser.add_argument(
        "--train_mode", 
        type=str, 
        default="", 
        help="Training mode configuration"
    )
    
    # 解码相关参数
    parser.add_argument(
        "--decode_truncate_len", 
        type=int, 
        default=50, 
        help="How many tokens to truncate from right"
    )
    parser.add_argument(
        "--decode_depth", 
        type=int, 
        default=2, 
        help="Decoding depth"
    )
    parser.add_argument(
        "--projection_top_p", 
        type=float, 
        default=0.2, 
        help="Top-p for projection sampling"
    )
    parser.add_argument(
        "--filter_top_p", 
        type=float, 
        default=1.0, 
        help="Top-p for filtering"
    )
    parser.add_argument(
        "--filter_top_p_prior", 
        type=float, 
        default=1.0, 
        help="Top-p for prior filtering"
    )

    # 模型推理参数
    parser.add_argument(
        "--big_model_inference", 
        type=str, 
        default="no",
        help="Whether to use big model inference mode"
    )

    # 自适应boost相关参数(新添加)
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1.0,
        help="Minimum boost delta value"
    )
    parser.add_argument(
        "--max_delta",
        type=float,
        default=10.0,
        help="Maximum boost delta value"
    )
    parser.add_argument(
        "--min_temp",
        type=float,
        default=0.1,
        help="Minimum temperature value"
    )
    parser.add_argument(
        "--base_temp",
        type=float,
        default=1.0,
        help="Base temperature value"
    )
    parser.add_argument(
        "--adaptive_mode",
        type=str,
        default="both",
        choices=["delta_only", "temp_only", "both", "none"],
        help="Which adaptive mechanism to use: delta_only, temp_only, both, or none"
    )
    parser.add_argument(
        "--jsd_threshold",
        type=float,
        default=0.0,
        help="JSD threshold below which minimum delta is used"
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.0,
        help="Entropy threshold below which minimum temperature is used"
    )
    parser.add_argument(
        "--fixed_delta",
        type=float,
        default=2.0,
        help="Fixed delta value when not using adaptive delta"
    )
    parser.add_argument(
        "--fixed_temp",
        type=float,
        default=1.0,
        help="Fixed temperature value when not using adaptive temperature"
    )
    parser.add_argument(
        "--stats_logging",
        action="store_true",
        help="Whether to log detailed statistics during generation"
    )

    # 输出配置参数(新添加)
    parser.add_argument(
        "--output_stats",
        action="store_true",
        help="Whether to include statistics in output files"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json"],
        help="Output file format"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="",
        help="Prefix for output filenames"
    )

    args = parser.parse_args()

    return args

def get_small_model_name(original_model_name):
    """将大模型映射到对应的小模型用于测试"""
    model_mapping = {
        "huggyllama/llama-7b": "huggyllama/llama-7b",  # 7B -> 125M "facebook/opt-125m"
        "huggyllama/llama-13b": "facebook/opt-350m",  # 13B -> 350M "facebook/opt-350m"
        # 可以添加更多映射
    }
    return model_mapping.get(original_model_name, "facebook/opt-125m")  # 默认使用 opt-125m

def main():
    args = parse_args()

    # 参数验证
    if args.adaptive_mode not in ["delta_only", "temp_only", "both", "none"]:
        raise ValueError(f"Invalid adaptive_mode: {args.adaptive_mode}")
        
    if args.min_delta > args.max_delta:
        raise ValueError(f"min_delta ({args.min_delta}) cannot be greater than max_delta ({args.max_delta})")
        
    if args.min_temp > args.base_temp:
        raise ValueError(f"min_temp ({args.min_temp}) cannot be greater than base_temp ({args.base_temp})")
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    cache_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    args.file_mode = args.file_mode.split('|')
    assert args.file_mode[0] == "fin"
    assert os.path.exists(args.file_mode[1])
    fin_path = args.file_mode[1]
    fin_data = []
    with open(fin_path, 'r', encoding='utf-8') as f:
        for line in f:
            proc_line = line.strip()
            if proc_line:
                fin_data.append(json.loads(proc_line))

    first_data = fin_data[0]
    original_model_name = first_data['assigned_model'].split('|')[0]
    args.model_name_or_path = get_small_model_name(original_model_name)
    logger.info(f"Original model: {original_model_name}")
    logger.info(f"Using small model for testing: {args.model_name_or_path}")

    try:
        logger.info("Loading tokenizer...")
        if 'llama' in args.model_name_or_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                use_fast=True,
                cache_dir=cache_dir,
                use_auth_token="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"
            )
        elif 'opt' in args.model_name_or_path.lower():
            tokenizer = GPT2Tokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=cache_dir,
                use_auth_token="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"
            )
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                cache_dir=cache_dir,
                use_fast=not args.use_slow_tokenizer,
                use_auth_token="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"
            )
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model...")
        if 'llama' in args.model_name_or_path.lower():
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                device_map='auto',
                cache_dir=cache_dir,
                use_auth_token="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"
            )
        elif 'opt' in args.model_name_or_path.lower():
            model = OPTForCausalLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                use_auth_token="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"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                use_auth_token="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"
            )
        
        model = model.to(device)
        model.eval()

        args.is_llama = 'llama' in args.model_name_or_path.lower()
        args.model_category = 'causal'
        args.vocab_size = model.config.vocab_size
        args.hidden_size = model.config.hidden_size
        args.one_hot_value = 5.0
        args.tokenizer = tokenizer
            
        export_list = []
        args.orig_decode_truncate_len = args.decode_truncate_len
        
        with torch.no_grad():
            for _fd in tqdm(fin_data, desc="Processing inputs"):
                try:
                    args.assigned_weight = _fd.get('assigned_weight', 1.0)
                    args.filter_top_p = _fd.get('filter_p', getattr(args, 'filter_top_p', 1.0))
                    args.filter_top_p_prior = _fd.get('filter_p_prior', getattr(args, 'filter_top_p_prior', 1.0))

                    ctx_field_name = 'context_string'
                    assert ctx_field_name in _fd
                    
                    input_ids = torch.tensor(
                        tokenizer.encode(_fd[ctx_field_name], add_special_tokens=True),
                        dtype=torch.long
                    ).unsqueeze(0).to(device)
                    
                    args.context_size = input_ids.size(1)
                    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
                    
                    if args.decode_truncate_len < 0:
                        logger.warning(f"Skipping long input {_fd['input_index']}")
                        continue

                    with torch.cuda.amp.autocast():
                        history_decode_ids, init_context_ids, avg_stats, sampled_sequences, context_sequences, stats = \
                            decode(args, input_ids, args.decode_depth, model, tokenizer)

                    export_dict = {
                        'tokens': [history_decode_ids.tolist()[0]],
                        'string': [sampled_sequences[0]],
                        'input_index': _fd['input_index'],
                        'output_index': len(export_list),
                        'assigned_model': args.model_name_or_path,
                        'original_model': original_model_name,
                        'assigned_weight': _fd.get('assigned_weight', 1.0),
                        'assigned_process': 0,
                        'avg_delta': avg_stats['avg_delta'],
                        'avg_temperature': avg_stats['avg_temperature'],
                        'avg_jsd': avg_stats['avg_jsd'],
                        'avg_entropy': avg_stats['avg_entropy'],
                        'context': context_sequences[0] if context_sequences else None,
                        'detailed_stats': stats
                    }
                    export_list.append(export_dict)

                except Exception as e:
                    logger.error(f"Error processing input {_fd['input_index']}: {str(e)}")
                    logger.error("Error details:", exc_info=True)
                    continue

        output_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/output/llama7b"
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.basename(fin_path)
        out_json_fn = f"{base_filename}.output_topp{args.projection_top_p}_genlen{args.decode_depth}_adaptive.jsonl"
        out_json_fn = os.path.join(output_dir, out_json_fn)
        os.makedirs(os.path.dirname(out_json_fn), exist_ok=True)
        
        with open(out_json_fn, mode="w") as f_out:
            for export in export_list:
                f_out.write(json.dumps(export))
                f_out.write("\n")

        logger.info(f"Successfully processed {len(export_list)} out of {len(fin_data)} inputs")
        logger.info(f"Results saved to {out_json_fn}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error("Error details:", exc_info=True)
        raise

if __name__ == "__main__":
    main()