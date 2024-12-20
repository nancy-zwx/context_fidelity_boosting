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



# 1. 添加新的辅助函数
def get_context_token_ids(context_input_ids):
    """获取context中的unique token ids"""
    return torch.unique(context_input_ids)

def create_boost_mask(logits, context_tokens, delta):
    """创建boost mask用于增强context tokens的权重"""
    boost_mask = torch.zeros_like(logits)
    for token in context_tokens:
        boost_mask[..., token] = delta
    return boost_mask


# watermark-based decode function
# def decode(args, batch_input_ids, dec_depth, model, tokenizer):
#     batch_size = args.per_device_eval_batch_size
#     assert batch_input_ids.size(1) == args.context_size
#     assert args.decode_truncate_len >= 0
#     assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
#     unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)

#     # 获取context input
#     if args.context_size > 0:
#         unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
#         # 获取context中的unique tokens
#         context_tokens = get_context_token_ids(unit_context_input_ids[0])
#     else:
#         raise ValueError("context cannot be none")
    
#     history_decode_ids = None
#     past_key_values = None

#     # 设置boost参数
#     delta = args.context_boost_delta if hasattr(args, 'context_boost_delta') else 2.0  # 默认boost值

#     if args.model_category == 'seq2seq':
#         model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
#             batch_input_ids[:, :args.context_size].clone(), dict(), None
#         )
#         history_decode_ids = model._prepare_decoder_input_ids_for_generation(
#             batch_input_ids.size(0),
#             model_kwargs=model_kwargs,
#             device=batch_input_ids.device,
#         )
#     else:
#         model_kwargs = None

#     for _i in range(dec_depth):
#         if args.model_category == 'causal':
#             model_inputs = model.prepare_inputs_for_generation(unit_context_input_ids, past_key_values=past_key_values)
#             outputs = model(**model_inputs, output_hidden_states=False)
#         elif args.model_category == 'seq2seq':
#             model_inputs = model.prepare_inputs_for_generation(history_decode_ids, **model_kwargs)
#             outputs = model(**model_inputs, output_hidden_states=False)
#         else:
#             raise ValueError("model category not supported")

#         # 获取logits
#         logits = outputs.logits[:, -1:, :].clone().contiguous()
        
#         # 创建并应用boost mask
#         boost_mask = create_boost_mask(logits, context_tokens, delta)
#         enhanced_logits = logits + boost_mask

#         # 应用过滤
#         if args.assigned_weight >= 0:
#             score = filter_logits_top_p(enhanced_logits, top_p=args.filter_top_p)
#         else:
#             score = filter_logits_top_p(enhanced_logits, top_p=args.filter_top_p_prior, negative_multiplier=True)

#         # 采样
#         projected_logits = logits_sampling_projection(score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
        
#         # 计算token IDs
#         simplex = torch.nn.functional.softmax(projected_logits, dim=-1)
#         real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)

#         # 更新历史
#         if args.model_category == 'causal':
#             unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)

#         if history_decode_ids is None:
#             history_decode_ids = real_token_ids_list
#         else:
#             history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

#         if args.model_category == 'causal':
#             past_key_values = outputs.past_key_values
#         elif args.model_category == 'seq2seq':
#             model_kwargs["past_key_values"] = outputs.past_key_values

#         # 检查是否生成结束
#         if real_token_ids_list[0][-1] == model.generation_config.eos_token_id:
#             break

#     # 处理输出
#     if args.context_size > 0:
#         init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
#         context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
#     else:
#         init_context_input_ids = None
#         context_sequences = None

#     sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'), skip_special_tokens=True)
#     logger.info(f"context: {context_sequences}")
#     logger.info(f"sampled: {colored(str(sampled_sequences), 'red')}")

#     return history_decode_ids, init_context_input_ids, None, sampled_sequences, context_sequences, None


# adjust for llama
def decode(args, batch_input_ids, dec_depth, model, tokenizer):
    batch_size = args.per_device_eval_batch_size
    assert batch_input_ids.size(1) == args.context_size
    assert args.decode_truncate_len >= 0
    assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)

    # 获取context input
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        # 获取context中的unique tokens
        context_tokens = get_context_token_ids(unit_context_input_ids[0])
    else:
        raise ValueError("context cannot be none")
    
    history_decode_ids = None
    past_key_values = None

    # 设置boost参数
    delta = args.context_boost_delta if hasattr(args, 'context_boost_delta') else 2.0  # 默认boost值

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
        # 修改这部分，添加llama的特殊处理
        if args.model_category == 'causal':
            if 'llama' in args.model_name_or_path.lower():
                # llama模型特殊处理
                if past_key_values is not None:
                    model_inputs = {
                        "input_ids": unit_context_input_ids[:, -1:],
                        "attention_mask": torch.ones_like(unit_context_input_ids),
                        "past_key_values": past_key_values,
                        "use_cache": True
                    }
                else:
                    model_inputs = {
                        "input_ids": unit_context_input_ids,
                        "attention_mask": torch.ones_like(unit_context_input_ids),
                        "use_cache": True
                    }
            else:
                # 原有的处理逻辑
                model_inputs = model.prepare_inputs_for_generation(
                    unit_context_input_ids, 
                    past_key_values=past_key_values
                )
        elif args.model_category == 'seq2seq':
            model_inputs = model.prepare_inputs_for_generation(
                history_decode_ids, 
                **model_kwargs
            )
        else:
            raise ValueError("model category not supported")

        outputs = model(**model_inputs, output_hidden_states=False)

        # 获取logits
        logits = outputs.logits[:, -1:, :].clone().contiguous()
        
        # 创建并应用boost mask
        boost_mask = create_boost_mask(logits, context_tokens, delta)
        enhanced_logits = logits + boost_mask

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

        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

        if args.model_category == 'causal':
            past_key_values = outputs.past_key_values
        elif args.model_category == 'seq2seq':
            model_kwargs["past_key_values"] = outputs.past_key_values

        # 检查是否生成结束
        if real_token_ids_list[0][-1] == model.generation_config.eos_token_id:
            break

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

    return history_decode_ids, init_context_input_ids, None, sampled_sequences, context_sequences, None

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")

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
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
    parser.add_argument("--init_blank_language_model", action="store_true", help="Whether or not to use a completely blank LM.")
    parser.add_argument(
        "--file_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--train_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--decode_truncate_len", type=int, default=50, help="",
    ) # how many to cut from right
    parser.add_argument(
        "--decode_depth", type=int, default=2, help="",
    )
    parser.add_argument(
        "--projection_top_p", type=float, default=0.2, help="",
    )
    parser.add_argument(
        "--filter_top_p", type=float, default=1.0, help="",
    )
    parser.add_argument(
        "--filter_top_p_prior", type=float, default=1.0, help="",
    )
    parser.add_argument("--big_model_inference", type=str, default="no")

    # 添加context boost相关参数
    parser.add_argument(
        "--context_boost_delta",
        type=float,
        default=2.0,
        help="Boost value for context tokens",
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


# adjusted main function for watermark decoding
# def main():
#     args = parse_args()

#     # 1. 基础设置
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger = logging.getLogger(__name__)

#     # 2. 设置环境变量和缓存路径
#     cache_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
#     os.makedirs(cache_dir, exist_ok=True)
#     os.environ['TRANSFORMERS_CACHE'] = cache_dir
#     os.environ['HF_HOME'] = cache_dir

#     # 3. 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if torch.cuda.is_available():
#         torch.cuda.set_device(0)

#     # 4. 设置随机种子
#     if args.seed is not None:
#         torch.manual_seed(args.seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(args.seed)

#     # 5. 读取输入文件
#     args.file_mode = args.file_mode.split('|')
#     assert args.file_mode[0] == "fin"
#     assert os.path.exists(args.file_mode[1])
#     fin_path = args.file_mode[1]
#     fin_data = []
#     with open(fin_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             proc_line = line.strip()
#             if proc_line:
#                 fin_data.append(json.loads(proc_line))

#     # 6. 获取模型名称并映射
#     first_data = fin_data[0]
#     original_model_name = first_data['assigned_model'].split('|')[0]
#     args.model_name_or_path = get_small_model_name(original_model_name)
#     logger.info(f"Original model: {original_model_name}")
#     logger.info(f"Using small model for testing: {args.model_name_or_path}")

#     try:
#         # 7. 加载tokenizer
#         logger.info("Loading tokenizer...")
#         if 'opt' in args.model_name_or_path.lower():
#             tokenizer = GPT2Tokenizer.from_pretrained(
#                 args.model_name_or_path,
#                 cache_dir=cache_dir,
#                 use_auth_token="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"
#             )
#             tokenizer.pad_token = tokenizer.eos_token
#         else:
#             tokenizer = AutoTokenizer.from_pretrained(
#                 args.model_name_or_path,
#                 cache_dir=cache_dir,
#                 use_fast=not args.use_slow_tokenizer,
#                 use_auth_token="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"
#             )
#         logger.info("Tokenizer loaded successfully")

#         # 8. 加载模型
#         logger.info("Loading model...")
#         if 'opt' in args.model_name_or_path.lower():
#             model = OPTForCausalLM.from_pretrained(
#                 args.model_name_or_path,
#                 cache_dir=cache_dir,
#                 torch_dtype=torch.float16,
#                 use_auth_token="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"
#             )
#         else:
#             model = AutoModelForCausalLM.from_pretrained(
#                 args.model_name_or_path,
#                 cache_dir=cache_dir,
#                 torch_dtype=torch.float16,
#                 use_auth_token="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"
#             )
        
#         model = model.to(device)
#         model.eval()
#         logger.info("Model loaded successfully and moved to GPU")

#         # 9. 设置模型参数
#         args.model_category = 'causal'
#         args.vocab_size = model.config.vocab_size
#         args.hidden_size = model.config.hidden_size
#         args.one_hot_value = 5.0
#         args.tokenizer = tokenizer

#         # 设置context boost相关参数
#         if not hasattr(args, 'context_boost_delta'):
#             args.context_boost_delta = 2.0  # 默认boost值
            
#         # 10. 处理生成
#         export_list = []
#         args.orig_decode_truncate_len = args.decode_truncate_len
        
#         with torch.no_grad():
#             for _fd in tqdm(fin_data, desc="Processing inputs"):
#                 try:
#                     # 获取参数
#                     args.assigned_weight = _fd.get('assigned_weight', 1.0)
#                     args.filter_top_p = _fd.get('filter_p', getattr(args, 'filter_top_p', 1.0))
#                     args.filter_top_p_prior = _fd.get('filter_p_prior', getattr(args, 'filter_top_p_prior', 1.0))

#                     # 处理输入
#                     ctx_field_name = 'context_string'
#                     assert ctx_field_name in _fd
                    
#                     input_ids = torch.tensor(
#                         tokenizer.encode(_fd[ctx_field_name], add_special_tokens=True),
#                         dtype=torch.long
#                     ).unsqueeze(0).to(device)
                    
#                     args.context_size = input_ids.size(1)  # 确保设置context_size
#                     args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
                    
#                     if args.decode_truncate_len < 0:
#                         logger.warning(f"Skipping long input {_fd['input_index']}")
#                         continue

#                     # 生成文本 - 使用新的watermark-based decode
#                     with torch.cuda.amp.autocast():
#                         history_decode_ids, init_context_ids, _, sampled_sequences, context_sequences, _ = \
#                             decode(args, input_ids, args.decode_depth, model, tokenizer)

#                     # 保存结果 - 添加更多信息
#                     export_dict = {
#                         'tokens': [history_decode_ids.tolist()[0]],
#                         'string': [sampled_sequences[0]],
#                         'input_index': _fd['input_index'],
#                         'output_index': len(export_list),
#                         'assigned_model': args.model_name_or_path,
#                         'original_model': original_model_name,
#                         'assigned_weight': _fd.get('assigned_weight', 1.0),
#                         'assigned_process': 0,
#                         'context_boost_delta': args.context_boost_delta,  # 添加boost参数信息
#                         'context': context_sequences[0] if context_sequences else None,
#                     }
#                     export_list.append(export_dict)
#                     logger.info(f"Processed input {_fd['input_index']}")
#                     logger.info(f"Context boost delta: {args.context_boost_delta}")

#                 except Exception as e:
#                     logger.error(f"Error processing input {_fd['input_index']}: {str(e)}")
#                     logger.error("Error details:", exc_info=True)
#                     continue

#         # 11. 保存结果 - 修改文件名以包含boost信息

#         output_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/output"
#         os.makedirs(output_dir, exist_ok=True)
#         base_filename = os.path.basename(fin_path)
#         out_json_fn = f"{base_filename}.output_topp{args.projection_top_p}_genlen{args.decode_depth}_boost{args.context_boost_delta}.jsonl"
#         out_json_fn = os.path.join(output_dir,out_json_fn)
#         with open(out_json_fn, mode="w") as f_out:
#             for export in export_list:
#                 f_out.write(json.dumps(export))
#                 f_out.write("\n")

#         logger.info(f"Successfully processed {len(export_list)} out of {len(fin_data)} inputs")
#         logger.info(f"Results saved to {out_json_fn}")


#     except Exception as e:
#         logger.error(f"Fatal error: {str(e)}")
#         logger.error("Error details:", exc_info=True)
#         raise



## adjust for llama
def main():
    args = parse_args()

    # 1. 基础设置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # 2. 设置环境变量和缓存路径
    cache_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir

    # 3. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # 4. 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # 5. 读取输入文件
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

    # 6. 获取模型名称并映射
    first_data = fin_data[0]
    original_model_name = first_data['assigned_model'].split('|')[0]
    args.model_name_or_path = get_small_model_name(original_model_name)
    logger.info(f"Original model: {original_model_name}")
    logger.info(f"Using small model for testing: {args.model_name_or_path}")

    try:
        # 7. 加载tokenizer
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
        logger.info("Tokenizer loaded successfully")

        # 8. 加载模型
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
        logger.info("Model loaded successfully and moved to GPU")

        # 在模型参数设置部分添加模型类型标记
        args.is_llama = 'llama' in args.model_name_or_path.lower()

        # 9. 设置模型参数
        args.model_category = 'causal'
        args.vocab_size = model.config.vocab_size
        args.hidden_size = model.config.hidden_size
        args.one_hot_value = 5.0
        args.tokenizer = tokenizer

        # 设置context boost相关参数
        if not hasattr(args, 'context_boost_delta'):
            args.context_boost_delta = 2.0  # 默认boost值
            
        # 10. 处理生成
        export_list = []
        args.orig_decode_truncate_len = args.decode_truncate_len
        
        with torch.no_grad():
            for _fd in tqdm(fin_data, desc="Processing inputs"):
                try:
                    # 获取参数
                    args.assigned_weight = _fd.get('assigned_weight', 1.0)
                    args.filter_top_p = _fd.get('filter_p', getattr(args, 'filter_top_p', 1.0))
                    args.filter_top_p_prior = _fd.get('filter_p_prior', getattr(args, 'filter_top_p_prior', 1.0))

                    # 处理输入
                    ctx_field_name = 'context_string'
                    assert ctx_field_name in _fd
                    
                    input_ids = torch.tensor(
                        tokenizer.encode(_fd[ctx_field_name], add_special_tokens=True),
                        dtype=torch.long
                    ).unsqueeze(0).to(device)
                    
                    args.context_size = input_ids.size(1)  # 确保设置context_size
                    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
                    
                    if args.decode_truncate_len < 0:
                        logger.warning(f"Skipping long input {_fd['input_index']}")
                        continue

                    # 生成文本 - 使用新的watermark-based decode
                    with torch.cuda.amp.autocast():
                        history_decode_ids, init_context_ids, _, sampled_sequences, context_sequences, _ = \
                            decode(args, input_ids, args.decode_depth, model, tokenizer)

                    # 保存结果 - 添加更多信息
                    export_dict = {
                        'tokens': [history_decode_ids.tolist()[0]],
                        'string': [sampled_sequences[0]],
                        'input_index': _fd['input_index'],
                        'output_index': len(export_list),
                        'assigned_model': args.model_name_or_path,
                        'original_model': original_model_name,
                        'assigned_weight': _fd.get('assigned_weight', 1.0),
                        'assigned_process': 0,
                        'context_boost_delta': args.context_boost_delta,  # 添加boost参数信息
                        'context': context_sequences[0] if context_sequences else None,
                    }
                    export_list.append(export_dict)
                    logger.info(f"Processed input {_fd['input_index']}")
                    logger.info(f"Context boost delta: {args.context_boost_delta}")

                except Exception as e:
                    logger.error(f"Error processing input {_fd['input_index']}: {str(e)}")
                    logger.error("Error details:", exc_info=True)
                    continue

        # 11. 保存结果 - 修改文件名以包含boost信息

        output_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/cfb/output/llama7b"
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.basename(fin_path)
        out_json_fn = f"{base_filename}.output_topp{args.projection_top_p}_genlen{args.decode_depth}_boost{args.context_boost_delta}.jsonl"
        out_json_fn = os.path.join(output_dir,out_json_fn)
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
