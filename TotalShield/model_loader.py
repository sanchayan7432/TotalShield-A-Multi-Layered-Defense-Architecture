# File: model_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_secured_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ✅ Add this block to ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return SecureLLMWrapper(model, tokenizer)



# def load_secured_model(model_name):
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     return SecureLLMWrapper(model, tokenizer)

# class SecureLLMWrapper:
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer

#     def generate(self, prompt):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#         outputs = self.model.generate(**inputs, max_new_tokens=100)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class SecureLLMWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        if pad_token_id is None:
            # Set to eos_token_id as fallback
            pad_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.convert_tokens_to_ids("<pad>")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
