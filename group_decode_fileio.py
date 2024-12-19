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

# original decode function
# def decode(args, batch_input_ids, dec_depth, model, tokenizer):
#     batch_size = args.per_device_eval_batch_size
#     assert batch_input_ids.size(1) == args.context_size
#     assert args.decode_truncate_len >= 0
#     assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
#     unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
#     if args.context_size > 0:
#         unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
#     else:
#         raise ValueError("context cannot be none")
#     history_decode_ids = None

#     past_key_values = None # necessary for causal models
#     if args.model_category == 'seq2seq':
#         model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
#             batch_input_ids[:, :args.context_size].clone(), dict(), None
#         ) # this step includes encoding the context
#         history_decode_ids = model._prepare_decoder_input_ids_for_generation(
#             batch_input_ids.size(0),
#             model_kwargs=model_kwargs,
#             device=batch_input_ids.device,
#         ) # create placeholder starter seq for decoding
#     else:
#         model_kwargs = None

#     for _i in range(dec_depth):
#         if args.model_category == 'causal':
#             model_inputs = model.prepare_inputs_for_generation(unit_context_input_ids, past_key_values=past_key_values)
#             outputs = model(**model_inputs, output_hidden_states=False)
#         elif args.model_category == 'seq2seq':
#             model_inputs = model.prepare_inputs_for_generation(history_decode_ids, **model_kwargs) # this incorporates past_key_values
#             outputs = model(**model_inputs, output_hidden_states=False)
#         else:
#             raise ValueError("model category not supported")

#         score = outputs.logits[:, -1:, :].clone().contiguous()

#         if args.assigned_weight >= 0:
#             score = filter_logits_top_p(score, top_p=args.filter_top_p)
#         else:
#             score = filter_logits_top_p(score, top_p=args.filter_top_p_prior, negative_multiplier=True)

#         score = args.assigned_weight * score
#         torch.distributed.all_reduce(score, group=args.gathering_group)

#         projected_logits = logits_sampling_projection(score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)

#         if not args.accelerator.is_main_process:
#             projected_logits = torch.zeros_like(projected_logits)
#         torch.distributed.all_reduce(projected_logits, group=args.gathering_group)

#         simplex = torch.nn.functional.softmax(projected_logits, dim=-1)
#         real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)

#         if args.model_category == 'causal':
#             unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1) # not really necessary but keeping

#         if history_decode_ids is None:
#             history_decode_ids = real_token_ids_list
#         else:
#             history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

#         if args.model_category == 'causal':
#             past_key_values = outputs.past_key_values
#         elif args.model_category == 'seq2seq':
#             model_kwargs["past_key_values"] = outputs.past_key_values

#         # HACK: stop when seeing eos token, but asserting batch size is 1, unit_seq_len is 1, optimize later
#         assert real_token_ids_list.size(0) == 1
#         assert real_token_ids_list.size(1) == 1
#         if real_token_ids_list[0][-1] == model.generation_config.eos_token_id:
#             break

#     if args.context_size > 0:
#         init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
#         context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))#, skip_special_tokens=True)
#     else:
#         init_context_input_ids = None
#         context_sequences = None
#     sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'), skip_special_tokens=True)
#     logger.info(f"context: {context_sequences}")
#     logger.info(f"sampled: {colored(str(sampled_sequences), 'red')}")

#     return history_decode_ids, init_context_input_ids, None, sampled_sequences, context_sequences, None


# adjusted decode function on single gpu
# def decode(args, batch_input_ids, dec_depth, model, tokenizer):
#     # 保持原有的参数检查
#     batch_size = args.per_device_eval_batch_size
#     assert batch_input_ids.size(1) == args.context_size
#     assert args.decode_truncate_len >= 0
#     assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
#     unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)

#     # 保持原有的输入处理
#     if args.context_size > 0:
#         unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
#     else:
#         raise ValueError("context cannot be none")
    
#     history_decode_ids = None
#     past_key_values = None  # necessary for causal models

#     # 保持原有的模型类别处理
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

#         score = outputs.logits[:, -1:, :].clone().contiguous()

#         # 应用过滤，保持原有逻辑
#         if args.assigned_weight >= 0:
#             score = filter_logits_top_p(score, top_p=args.filter_top_p)
#         else:
#             score = filter_logits_top_p(score, top_p=args.filter_top_p_prior, negative_multiplier=True)

#         # 移除分布式相关代码，直接使用本地分数
#         projected_logits = logits_sampling_projection(score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
        
#         # 计算token IDs
#         simplex = torch.nn.functional.softmax(projected_logits, dim=-1)
#         real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)

#         # 更新历史记录
#         if args.model_category == 'causal':
#             unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)

#         if history_decode_ids is None:
#             history_decode_ids = real_token_ids_list
#         else:
#             history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

#         # 更新past_key_values
#         if args.model_category == 'causal':
#             past_key_values = outputs.past_key_values
#         elif args.model_category == 'seq2seq':
#             model_kwargs["past_key_values"] = outputs.past_key_values

#         # 保持原有的EOS检查
#         assert real_token_ids_list.size(0) == 1
#         assert real_token_ids_list.size(1) == 1
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

## adjusted for llama
def decode(args, batch_input_ids, dec_depth, model, tokenizer):
    batch_size = args.per_device_eval_batch_size
    assert batch_input_ids.size(1) == args.context_size
    assert args.decode_truncate_len >= 0
    assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)

    unit_context_input_ids = batch_input_ids[:, :args.context_size].clone() if args.context_size > 0 else None
    if unit_context_input_ids is None:
        raise ValueError("context cannot be none")
    
    history_decode_ids = None
    past_key_values = None
    attention_mask = torch.ones_like(unit_context_input_ids)

    for _i in range(dec_depth):
        # 针对不同模型类型调整输入准备
        if 'llama' in args.model_name_or_path.lower():
            if past_key_values is not None:
                # 对于已有历史的情况
                model_inputs = {
                    "input_ids": unit_context_input_ids[:, -1:],
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True
                }
            else:
                # 首次运行的情况
                model_inputs = {
                    "input_ids": unit_context_input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": True
                }
        else:
            try:
                model_inputs = model.prepare_inputs_for_generation(
                    unit_context_input_ids, 
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True
                )
            except:
                model_inputs = {
                    "input_ids": unit_context_input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True
                }

        outputs = model(**model_inputs, output_hidden_states=False)
        score = outputs.logits[:, -1:, :].clone().contiguous()

        if args.assigned_weight >= 0:
            score = filter_logits_top_p(score, top_p=args.filter_top_p)
        else:
            score = filter_logits_top_p(score, top_p=args.filter_top_p_prior, negative_multiplier=True)

        projected_logits = logits_sampling_projection(score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
        simplex = torch.nn.functional.softmax(projected_logits, dim=-1)
        real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)

        # 更新状态
        unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
        attention_mask = torch.ones_like(unit_context_input_ids)
        past_key_values = outputs.past_key_values

        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

        # EOS检查
        if real_token_ids_list.size(0) == 1 and real_token_ids_list.size(1) == 1:
            if real_token_ids_list[0][-1] == model.generation_config.eos_token_id:
                break

    # 输出处理部分保持不变
    context_sequences = None
    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().cpu())
    else:
        init_context_input_ids = None

    sampled_sequences = tokenizer.batch_decode(history_decode_ids.detach().cpu(), skip_special_tokens=True)
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
    args = parser.parse_args()

    return args

# original main function
# def main():

#     args = parse_args()

#     accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=259200))])
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger.info(accelerator.state)

#     logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
#     if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
#         transformers.utils.logging.set_verbosity_info()
#     else:
#         datasets.utils.logging.set_verbosity_error()
#         transformers.utils.logging.set_verbosity_error()

#     if args.seed is not None:
#         # set_seed(args.seed)
#         accelerate.utils.set_seed(args.seed, device_specific=True) # differ slightly for each device

#     if accelerator.is_main_process:
#         pass
#         # if args.output_dir is not None:
#         #     os.makedirs(args.output_dir, exist_ok=True)
#     accelerator.wait_for_everyone()

#     if args.train_mode == "decode":
#         if len(args.model_name_or_path.split('|')) > 0:
#             main_model_name = args.model_name_or_path.split('|')[0]
#             fallback_model_name = args.model_name_or_path.split('|')[1]
#             args.model_name_or_path = main_model_name
#             args.orig_model_name_or_path = fallback_model_name
#         else:
#             args.orig_model_name_or_path = args.model_name_or_path
#     else:
#         raise ValueError("training should be in a separate file (irrelevant in context-aware decoding)")

#     # Han: assign ensemble models
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
#     rank2model = dict()
#     for _fd in fin_data:
#         if _fd['assigned_process'] in rank2model: # sanity check
#             assert ' '.join(rank2model[_fd['assigned_process']]) == ' '.join(_fd['assigned_model'].split('|'))
#         else:
#             rank2model[_fd['assigned_process']] = _fd['assigned_model'].split('|') 

#     # Han: add gathering group
#     default_backend = torch.distributed.get_backend(torch.distributed.distributed_c10d._get_default_group())
#     args.gathering_group = torch.distributed.new_group(ranks=list(sorted(rank2model.keys())), backend=default_backend)

#     if accelerator.process_index not in rank2model.keys(): # Han: exit if not in the ensemble
#         return
#     args.model_name_or_path = rank2model[accelerator.process_index][0]

#     if args.config_name:
#         config = AutoConfig.from_pretrained(args.config_name)
#     elif args.model_name_or_path:
#         if 'llama' in args.model_name_or_path.lower():
#             from transformers import LlamaConfig
#             config = LlamaConfig.from_pretrained(args.model_name_or_path)
#         else:
#             config = AutoConfig.from_pretrained(args.model_name_or_path)
#     else:
#         config = CONFIG_MAPPING[args.model_type]()
#         logger.warning("You are instantiating a new config instance from scratch.")

#     if 'neox' in args.model_name_or_path.lower(): # Han: gpt-neox doesn't have a slow tokenizer, use GPTNeoXTokenizerFast
#         from transformers import GPTNeoXTokenizerFast
#         tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.model_name_or_path)
#     elif 'llama' in args.model_name_or_path.lower():
#         from transformers import LlamaTokenizer
#         tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
#     else:
#         assert args.use_slow_tokenizer == True 
#         if args.tokenizer_name:
#             tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
#         elif args.model_name_or_path:
#             tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
#         else:
#             raise ValueError(
#                 "You are instantiating a new tokenizer from scratch. This is not supported by this script."
#                 "You can do it from another script, save it, and load it from here, using --tokenizer_name."
#             )

#     if args.init_blank_language_model:
#         raise ValueError("disabled")
#         model = AutoModelForMaskedLM.from_config(config)
#     elif args.model_name_or_path:
#         if 't5' in args.model_name_or_path.lower() or 'tk' in args.model_name_or_path.lower():
#             model = AutoModelForSeq2SeqLM.from_pretrained(
#                 args.model_name_or_path,
#                 from_tf=bool(".ckpt" in args.model_name_or_path),
#                 config=config,
#                 ignore_mismatched_sizes=False,
#                 torch_dtype=torch.float16,
#             )
#             args.model_category = 'seq2seq'
#             model = model.to(accelerator.device)
#         else:
#             if 'llama' in args.model_name_or_path.lower(): # llama special case
#                 from transformers import LlamaForCausalLM
#                 if args.big_model_inference == 'no':
#                     model = LlamaForCausalLM.from_pretrained(
#                         args.model_name_or_path,
#                         torch_dtype=torch.float16,
#                     )
#                     model = model.to(accelerator.device)
#                 else:
#                     # Han: we assume 8 GPUs
#                     if accelerator.process_index == 0:
#                         local_devices = [0, 2, 4, 6]
#                     elif accelerator.process_index == 1:
#                         local_devices = [1, 3, 5, 7]
#                     else:
#                         raise ValueError("check accelerator.process_index")
#                     # this is architecture specific
#                     my_device_map = {'model.embed_tokens': local_devices[0],
#                                     'lm_head': local_devices[0],
#                                     'model.norm': local_devices[0]}
#                     for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
#                         for layer_idx in layer_idx_list:
#                             my_device_map[f'model.layers.{layer_idx}'] = local_devices[_device_i]
#                     model = LlamaForCausalLM.from_pretrained(
#                         args.model_name_or_path,
#                         device_map=my_device_map,
#                         torch_dtype=torch.float16,
#                     )
#             elif args.big_model_inference == 'no':
#                 model = AutoModelForCausalLM.from_pretrained(
#                     args.model_name_or_path,
#                     from_tf=bool(".ckpt" in args.model_name_or_path),
#                     config=config,
#                     ignore_mismatched_sizes=False,
#                     torch_dtype=torch.float16, 
#                 )
#                 model = model.to(accelerator.device)
#             elif args.big_model_inference == 'yes' and 'opt' in args.model_name_or_path.lower():
#                 # Han: we assume 8 GPUs
#                 if accelerator.process_index == 0:
#                     local_devices = [0, 2, 4, 6]
#                 elif accelerator.process_index == 1:
#                     local_devices = [1, 3, 5, 7]
#                 else:
#                     raise ValueError("check accelerator.process_index")
#                 # this is architecture specific
#                 my_device_map = {'model.decoder.embed_tokens': local_devices[0],
#                                 'lm_head': local_devices[0],
#                                 'model.decoder.embed_positions': local_devices[0],
#                                 'model.decoder.final_layer_norm': local_devices[0]}
#                 for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
#                     for layer_idx in layer_idx_list:
#                         my_device_map[f'model.decoder.layers.{layer_idx}'] = local_devices[_device_i]
#                 model = AutoModelForCausalLM.from_pretrained(
#                     args.model_name_or_path,
#                     from_tf=bool(".ckpt" in args.model_name_or_path),
#                     config=config,
#                     ignore_mismatched_sizes=False,
#                     device_map=my_device_map,
#                     torch_dtype=torch.float16,
#                 )
#             elif args.big_model_inference == 'yes' and 'neox' in args.model_name_or_path.lower():
#                 # Han: we assume 8 GPUs
#                 if accelerator.process_index == 0:
#                     local_devices = [0, 2, 4, 6]
#                 elif accelerator.process_index == 1:
#                     local_devices = [1, 3, 5, 7]
#                 else:
#                     raise ValueError("check accelerator.process_index")
#                 # this is architecture specific
#                 my_device_map = {'gpt_neox.embed_in': local_devices[0],
#                                 'embed_out': local_devices[0],
#                                 'gpt_neox.final_layer_norm': local_devices[0]}
#                 for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
#                     for layer_idx in layer_idx_list:
#                         my_device_map[f'gpt_neox.layers.{layer_idx}'] = local_devices[_device_i]
#                 model = AutoModelForCausalLM.from_pretrained(
#                     args.model_name_or_path,
#                     from_tf=bool(".ckpt" in args.model_name_or_path),
#                     config=config,
#                     ignore_mismatched_sizes=False,
#                     device_map=my_device_map,
#                     torch_dtype=torch.float16,
#                 )
#             elif args.big_model_inference == 'yes' and 'neo' in args.model_name_or_path.lower():
#                 # Han: we assume 8 GPUs
#                 if accelerator.process_index == 0:
#                     local_devices = [0, 2, 4, 6]
#                 elif accelerator.process_index == 1:
#                     local_devices = [1, 3, 5, 7]
#                 else:
#                     raise ValueError("check accelerator.process_index")
#                 # this is architecture specific
#                 my_device_map = {'transformer.wte': local_devices[0],
#                                 'lm_head': local_devices[0],
#                                 'transformer.wpe': local_devices[0],
#                                 'transformer.drop': local_devices[0],
#                                 'transformer.ln_f': local_devices[0]}
#                 for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
#                     for layer_idx in layer_idx_list:
#                         my_device_map[f'transformer.h.{layer_idx}'] = local_devices[_device_i]
#                 model = AutoModelForCausalLM.from_pretrained(
#                     args.model_name_or_path,
#                     from_tf=bool(".ckpt" in args.model_name_or_path),
#                     config=config,
#                     ignore_mismatched_sizes=False,
#                     device_map=my_device_map,
#                     torch_dtype=torch.float16,
#                 )
#             else:
#                 raise ValueError("check args.big_model_inference")

#             args.model_category = 'causal'
#         model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward) # referred to https://github.com/huggingface/accelerate/blob/38fd30e764ea87ef86e7d69fcba559c3605925b1/src/accelerate/accelerator.py#L1138
#         model.forward = accelerate.utils.convert_outputs_to_fp32(model.forward)
#     else:
#         raise ValueError("specify --init_blank_language_model")

#     model.resize_token_embeddings(len(tokenizer))

#     logger.info(f"model size: {sum(p.numel() for p in model.parameters())}")
#     vocab_size = model.get_input_embeddings().weight.size(0)
#     hidden_size = model.get_input_embeddings().weight.size(1)
#     one_hot_value = 5.0 # unused

#     ##########################################

#     # change the output file name later
#     out_json_fn = f"{fin_path}.output_topp{args.projection_top_p}_genlen{args.decode_depth}.jsonl"
#     if accelerator.is_main_process:
#         with open(out_json_fn, 'w') as f:
#             f.write('placeholder, program not finished ...\n')

#     args.tokenizer = tokenizer

#     if args.train_mode == "decode":
#         model.eval()

#         args.one_hot_value = one_hot_value
#         args.vocab_size = vocab_size
#         args.hidden_size = hidden_size
#         args.accelerator = accelerator

#         export_list = []
#         args.orig_decode_truncate_len = args.decode_truncate_len
#         with torch.no_grad():
#             for _fd in fin_data: # only support batch size 1 for now since the context size can be different across lines
#                 if _fd['assigned_process'] != args.accelerator.process_index: # remember to unblock barriers before this line
#                     continue
#                 args.assigned_weight = _fd['assigned_weight']

#                 ctx_field_name = 'context_string'
#                 assert ctx_field_name in _fd
#                 assert args.per_device_eval_batch_size == 1

#                 input_ids = torch.LongTensor(tokenizer.encode(_fd[ctx_field_name], add_special_tokens=True)).unsqueeze(0).to(args.accelerator.device)
#                 args.context_size = input_ids.size(1)
#                 args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size # Han: this compensates for the unknown input context size

#                 if 'filter_p' in _fd: # token filtering
#                     args.filter_top_p = _fd['filter_p']
#                 if 'filter_p_prior' in _fd:
#                     args.filter_top_p_prior = _fd['filter_p_prior']

#                 if args.decode_truncate_len < 0:
#                     continue # skipping very long examples
#                 logger.info(f"idx: {_fd['input_index']}")

#                 repeat_sample = 1 # change here manually if necessary
#                 for _r in range(repeat_sample):
#                     history_decode_ids, _, _, sampled_sequences, _, _ = \
#                         decode(args, input_ids, args.decode_depth, model, tokenizer)
#                     if _r == 0: # first sample
#                         # export to jsonl
#                         for _i in range(args.per_device_eval_batch_size):
#                             export_dict = dict()
#                             export_dict['tokens'] = [history_decode_ids.tolist()[_i]]
#                             export_dict['string'] = [sampled_sequences[_i]]
#                             export_dict['assigned_process'] = _fd['assigned_process']
#                             export_dict['assigned_model'] = args.model_name_or_path
#                             export_dict['output_index'] = len(export_list)
#                             export_dict['input_index'] = _fd['input_index']
#                             export_list.append(export_dict)
#                     else:
#                         for _i in range(args.per_device_eval_batch_size):
#                             export_list[-(args.per_device_eval_batch_size - _i)]['tokens'].append(history_decode_ids.tolist()[_i])
#                             export_list[-(args.per_device_eval_batch_size - _i)]['string'].append(sampled_sequences[_i])

#         if accelerator.is_main_process:
#             if os.path.exists(out_json_fn):
#                 os.remove(out_json_fn)
#                 logger.info(f"Cleaning existing {out_json_fn}")
#             with open(out_json_fn, mode="w") as f_out: # use mode 'a' if several processes are writing to the same file
#                 for export in export_list:
#                     f_out.write(json.dumps(export))
#                     f_out.write("\n")


def get_small_model_name(original_model_name):
    """将大模型映射到对应的小模型用于测试"""
    model_mapping = {
        "huggyllama/llama-7b": "huggyllama/llama-7b",  # 7B -> 125M "facebook/opt-125m"
        "huggyllama/llama-13b": "facebook/opt-350m",  # 13B -> 350M "facebook/opt-350m"
        # 可以添加更多映射
    }
    return model_mapping.get(original_model_name, "facebook/opt-125m")  # 默认使用 opt-125m


# # adjusted main function on single gpu
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
                    
#                     args.context_size = input_ids.size(1)
#                     args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
                    
#                     if args.decode_truncate_len < 0:
#                         logger.warning(f"Skipping long input {_fd['input_index']}")
#                         continue

#                     # 生成文本
#                     with torch.cuda.amp.autocast():
#                         history_decode_ids, _, _, sampled_sequences, _, _ = \
#                             decode(args, input_ids, args.decode_depth, model, tokenizer)

#                     # 保存结果
#                     export_dict = {
#                         'tokens': [history_decode_ids.tolist()[0]],
#                         'string': [sampled_sequences[0]],
#                         'input_index': _fd['input_index'],
#                         'output_index': len(export_list),
#                         'assigned_model': args.model_name_or_path,
#                         'original_model': original_model_name,
#                         'assigned_weight': _fd.get('assigned_weight', 1.0),
#                         'assigned_process': 0
#                     }
#                     export_list.append(export_dict)
#                     logger.info(f"Processed input {_fd['input_index']}")

#                 except Exception as e:
#                     logger.error(f"Error processing input {_fd['input_index']}: {str(e)}")
#                     logger.error("Error details:", exc_info=True)
#                     continue

#         # 11. 保存结果
#         output_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/context-aware-decoding/output"
#         os.makedirs(output_dir, exist_ok=True)
#         base_filename = os.path.basename(fin_path)
#         out_json_fn = f"{base_filename}.output_topp{args.projection_top_p}_genlen{args.decode_depth}.jsonl"
#         out_json_fn = os.path.join(output_dir,out_json_fn)


#         with open(out_json_fn, mode="w") as f_out:
#             for export in export_list:
#                 f_out.write(json.dumps(export))
#                 f_out.write("\n")

#         logger.info(f"Successfully processed {len(export_list)} out of {len(fin_data)} inputs")

#     except Exception as e:
#         logger.error(f"Fatal error: {str(e)}")
#         logger.error("Error details:", exc_info=True)
#         raise

# adjusted for llama
def main():
    args = parse_args()
    
    # 1. 基础设置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # 2. 环境设置
    cache_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.update({
        'TRANSFORMERS_CACHE': cache_dir,
        'HF_HOME': cache_dir
    })

    # 3. 设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None

    # 4. 读取输入数据
    args.file_mode = args.file_mode.split('|')
    assert args.file_mode[0] == "fin" and os.path.exists(args.file_mode[1])
    
    with open(args.file_mode[1], 'r', encoding='utf-8') as f:
        fin_data = [json.loads(line.strip()) for line in f if line.strip()]

    # 5. 模型配置
    first_data = fin_data[0]
    original_model_name = first_data['assigned_model'].split('|')[0]
    args.model_name_or_path = get_small_model_name(original_model_name)
    args.is_llama = 'llama' in args.model_name_or_path.lower()  # 添加模型类型检查
    logger.info(f"Original model: {original_model_name}")
    logger.info(f"Using small model: {args.model_name_or_path}")
    logger.info(f"Is LLaMA model: {args.is_llama}")

    try:
        # 6. 加载tokenizer和model
        model_loading_kwargs = {
            "cache_dir": cache_dir,
            "use_auth_token": "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"
        }

        # 根据模型类型选择tokenizer
        if args.is_llama:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                use_fast=True,
                **model_loading_kwargs
            )
        elif 'opt' in args.model_name_or_path.lower():
            tokenizer = GPT2Tokenizer.from_pretrained(
                args.model_name_or_path,
                **model_loading_kwargs
            )
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                use_fast=not getattr(args, 'use_slow_tokenizer', False),
                **model_loading_kwargs
            )
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

        # 根据模型类型加载模型
        if args.is_llama:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                device_map='auto',  # 添加设备映射
                **model_loading_kwargs
            )
        elif 'opt' in args.model_name_or_path.lower():
            model = OPTForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                **model_loading_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                **model_loading_kwargs
            )

        model = model.to(device)
        model.eval()

        # 7. 模型参数设置
        args_params = {
            'model_category': 'causal',
            'vocab_size': model.config.vocab_size,
            'hidden_size': model.config.hidden_size,
            'one_hot_value': 5.0,
            'tokenizer': tokenizer,
            'max_position_embeddings': getattr(model.config, 'max_position_embeddings', 2048)
        }
        for param_name, param_value in args_params.items():
            setattr(args, param_name, param_value)

        # 8. 生成处理
        export_list = []
        args.orig_decode_truncate_len = args.decode_truncate_len
        
        with torch.no_grad():
            for _fd in tqdm(fin_data, desc="Processing inputs"):
                try:
                    # 更新参数
                    for param_name, default_value in [
                        ('assigned_weight', 1.0),
                        ('filter_top_p', getattr(args, 'filter_top_p', 1.0)),
                        ('filter_top_p_prior', getattr(args, 'filter_top_p_prior', 1.0))
                    ]:
                        setattr(args, param_name, _fd.get(param_name.replace('filter_', 'filter_p'), default_value))

                    # 处理输入
                    input_text = _fd.get('context_string')
                    if not input_text:
                        continue

                    # 添加模型特定的输入处理
                    if args.is_llama:
                        tokens = tokenizer.encode(input_text, add_special_tokens=True, truncation=True, 
                                               max_length=args.max_position_embeddings)
                    else:
                        tokens = tokenizer.encode(input_text, add_special_tokens=True)

                    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
                    
                    args.context_size = input_ids.size(1)
                    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
                    
                    if args.decode_truncate_len < 0:
                        logger.warning(f"Skipping long input {_fd['input_index']}")
                        continue

                    # 生成文本
                    with torch.cuda.amp.autocast():
                        history_decode_ids, _, _, sampled_sequences, _, _ = \
                            decode(args, input_ids, args.decode_depth, model, tokenizer)

                    # 保存结果
                    export_list.append({
                        'tokens': [history_decode_ids.tolist()[0]],
                        'string': [sampled_sequences[0]],
                        'input_index': _fd['input_index'],
                        'output_index': len(export_list),
                        'assigned_model': args.model_name_or_path,
                        'original_model': original_model_name,
                        'assigned_weight': _fd.get('assigned_weight', 1.0),
                        'assigned_process': 0
                    })

                except Exception as e:
                    logger.error(f"Error processing input {_fd['input_index']}: {str(e)}")
                    logger.error("Error details:", exc_info=True)
                    continue

        # 9. 保存结果
        output_dir = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/context-aware-decoding/output"
        os.makedirs(output_dir, exist_ok=True)
        out_json_fn = os.path.join(
            output_dir,
            f"{os.path.basename(args.file_mode[1])}.output_topp{args.projection_top_p}_genlen{args.decode_depth}.jsonl"
        )

        with open(out_json_fn, 'w') as f_out:
            for export in export_list:
                f_out.write(json.dumps(export) + "\n")

        logger.info(f"Successfully processed {len(export_list)} out of {len(fin_data)} inputs")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error("Error details:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
