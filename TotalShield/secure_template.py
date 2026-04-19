# File: secure_template.py
def secure_prompt(prompt):
    safe_intro = "You are an AI assistant. Ensure user safety."
    return safe_intro + "\n" + prompt
