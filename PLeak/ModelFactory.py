import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ModelFactory():
    def __init__(self):
        self.MODEL_CONF = {}
        self._register_model_config('gptj', 'EleutherAI/gpt-j-6b', 50400)
        self._register_model_config('opt', r'D:\SANCHAYANghosh\hf_models\facebookopt-6.7B', 50272)
        self._register_model_config('llama', 'D:\SANCHAYANghosh\hf_models\llama-2-7b-hf', 32000)
        self._register_model_config('llama-70b', 'meta-llama/Llama-2-70b-chat-hf', 32000)
        self._register_model_config('llama-chat', 'meta-llama/Llama-2-7b-chat-hf', 32000)
        self._register_model_config('falcon', r'D:\SANCHAYANghosh\hf_models\falcon-7b', 65024)
        self._register_model_config('vicuna', 'D:\SANCHAYANghosh\hf_models\lmsysvicuna-7b-v1.5', 32000)
        self._register_model_config('tsm', 'D:\SANCHAYANghosh\TotalShield\TotalShieldModel', 50258)
        self._register_model_config('deepseekv3.2', 'deepseek-ai/DeepSeek-V3.2', 128000)
        self._register_model_config('deepseek7b', 'D:\SANCHAYANghosh\hf_models\deepseek-aideepseek-llm-7b-base', 100000)
        self._register_model_config('pleakgard', 'D:\SANCHAYANghosh\PLeakGard\pleakgard_model', 32000)
        self._register_model_config('msphi', 'D:\SANCHAYANghosh\hf_models\microsoftphi-2', 50295)

    def _register_model_config(self, name, alias, vocab_size):
        self.MODEL_CONF[name] = {'alias': alias, 'vocab_size': vocab_size}


    def get_vocab_size(self, name):
        return self.MODEL_CONF[name]['vocab_size']

    def get_tokenizer(self, name):
        return AutoTokenizer.from_pretrained(self.MODEL_CONF[name]['alias'])

    def get_model(self, name):
        # return AutoModelForCausalLM.from_pretrained(self.MODEL_CONF[name]['alias'], device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16).eval()
        return AutoModelForCausalLM.from_pretrained(
            self.MODEL_CONF[name]['alias'],
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        ).eval()