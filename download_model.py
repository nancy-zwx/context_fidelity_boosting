import os
import logging
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM, GPT2Tokenizer
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models_with_cache():
    """使用本地缓存加载模型"""
    try:
        # 打印版本信息
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # 设置具体的模型目录
        model_path = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"
        
        # 验证必要的文件是否存在
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'vocab.json']
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                raise ValueError(f"Required file {file} not found in {model_path}")
        
        logger.info(f"Loading from local directory: {model_path}")
        
        # 使用GPT2Tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        # 设置padding token
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        # 使用OPT专用的模型类
        logger.info("Loading model...")
        model = OPTForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16
        )
        logger.info("Model loaded successfully")
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to GPU")
            
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error("Error details:", exc_info=True)
        return None, None

def test_model_loading():
    try:
        # 加载模型
        logger.info("Starting model loading...")
        tokenizer, model = load_models_with_cache()
        
        if tokenizer is None or model is None:
            logger.error("Failed to load models")
            return False
            
        # 测试模型
        logger.info("Testing models...")
        test_text = "Hello, world!"
        
        # 测试tokenizer
        inputs = tokenizer(
            test_text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        logger.info(f"Tokenized input: {inputs}")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            logger.info("Inputs moved to GPU")
        
        # 测试模型推理
        with torch.no_grad():
            outputs = model(**inputs)
            logger.info("Model forward pass completed")
            
        logger.info("Model test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error("Error details:", exc_info=True)
        return False

if __name__ == "__main__":
    # 设置环境变量
    os.environ['TRANSFORMERS_CACHE'] = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
    os.environ['HF_HOME'] = "/apdcephfs_cq10/share_1567347/share_info/wendyzhang/.cache/huggingface"
    
    # 检查transformers版本
    logger.info(f"Transformers version: {transformers.__version__}")
    
    success = test_model_loading()
    if success:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Tests failed!")