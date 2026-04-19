# File: main_totalsecure.py
from model_loader import load_secured_model
from defense_totalsecure import TotalShieldDefense
from attack_simulator import PLeakAttacker
from evaluator import evaluate_attack_resistance
from secure_template import secure_prompt

if __name__ == '__main__':
    model = load_secured_model('mistralai/Mistral-7B-Instruct-v0.1')
    defender = TotalShieldDefense(model)
    attacker = PLeakAttacker(shadow_model=model)

    results = []
    for prompt in attacker.generate_prompts(num=500):
        safe_prompt = secure_prompt(prompt)
        response = defender.generate(safe_prompt)
        metrics = evaluate_attack_resistance(prompt, response)
        results.append(metrics)

    # Final evaluation
    avg_results = {
        k: sum([r[k] for r in results]) / len(results) for k in results[0]
    }
    print("\nFinal Metrics:")
    for k, v in avg_results.items():
        print(f"{k}: {v:.4f}")