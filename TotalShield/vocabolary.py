from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data1/SANCHAYANghosh01/TotalShield/TotalShieldModel")
print(len(tokenizer))  # prints vocab size, e.g. 50258
