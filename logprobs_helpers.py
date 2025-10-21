import requests
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer

def get_logprobs(prompt: str, vllm_url: str = "http://localhost:8000/v1/completions") -> Dict[str, Any]:
    """
    Get logprobs for an input prompt without generation.
    
    Args:
        prompt: The input text to get logprobs for
        vllm_url: URL of vLLM completions endpoint
        
    Returns:
        Dictionary containing logprobs information
    """
    payload = {
        "model": "Qwen/Qwen3-8b",
        "prompt": prompt,
        "max_tokens": 1,  # Minimal generation
        "temperature": 0.0,
        "logprobs": True,  # Request logprobs
        "echo": True,  # Echo the prompt tokens with their logprobs
        "stream": False
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(vllm_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        log_probs = []
        for token_dict in result['choices'][0]['prompt_logprobs'][1:]:
            for _, logprob_dict in token_dict.items():
                if (len(token_dict) == 1) or (logprob_dict['rank'] != 1):
                    log_probs.append(logprob_dict['logprob'])
        
        return log_probs
        
    except Exception as e:
        raise Exception(f"Failed to get logprobs: {e}")


def get_insertion_logprobs(prompt: str, insertion: str) -> List[float]:
    full_text = prompt + insertion
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b")
    n_prompt_tokens = len(tokenizer.tokenize(prompt)[1:])
    return get_logprobs(full_text)[n_prompt_tokens:]
