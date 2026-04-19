# File: evaluator.py
from utils import compute_exact_match, compute_substring_match, compute_edit_distance, compute_semantic_similarity

def evaluate_attack_resistance(prompt, response):
    return {
        'ExactMatch': compute_exact_match(prompt, response),
        'SubstringMatch': compute_substring_match(prompt, response),
        'EditDistance': compute_edit_distance(prompt, response),
        'SemanticSimilarity': compute_semantic_similarity(prompt, response)
    }