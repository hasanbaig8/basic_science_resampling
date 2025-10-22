#!/usr/bin/env python3
"""
FastAPI server for Voice-in-Head Pipeline

Exposes the Python pipeline functionality as REST endpoints for the web app.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from pipeline import RolloutGenerator, InterventionInserter, DecisionParser
from pipeline.voice_in_head_strategy import VoiceInHeadStrategy

# Initialize FastAPI app
app = FastAPI(
    title="Voice-in-Head Pipeline API",
    description="API for generating rollouts and interventions using the Python pipeline",
    version="1.0.0"
)

# Add CORS middleware to allow requests from React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for runpod port forwarding
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline components
rollout_generator = RolloutGenerator(max_tokens=10000)
voice_strategy = VoiceInHeadStrategy()
intervention_inserter = InterventionInserter(voice_strategy)
decision_parser = DecisionParser()


# Request/Response Models
class GenerateRolloutRequest(BaseModel):
    prompt: str
    n: int = 1
    max_tokens: int = 1000
    temperature: float = 0.7


class GenerateRolloutResponse(BaseModel):
    rollouts: List[str]


class GenerateInterventionsRequest(BaseModel):
    rollout: str
    goal_intervention: str
    original_prompt: str


class InterventionResultResponse(BaseModel):
    clipped_text: str
    generated_intervention: str
    all_good_interventions: List[str]
    selected_intervention_index: int


class ContinueFromInterventionRequest(BaseModel):
    original_prompt: str
    clipped_text: str
    generated_intervention: str


class ContinueFromInterventionResponse(BaseModel):
    continuation: str
    full_output: str


class GetLogprobsRequest(BaseModel):
    text: str


class TokenLogprob(BaseModel):
    token: str
    logprob: float


class GetLogprobsResponse(BaseModel):
    tokens_with_logprobs: List[TokenLogprob]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Voice-in-Head Pipeline API is running"}


@app.post("/generate-rollout", response_model=GenerateRolloutResponse)
async def generate_rollout(request: GenerateRolloutRequest):
    """
    Generate initial rollouts from a prompt.

    Uses RolloutGenerator to create completions.
    """
    print(f"\n[DEBUG] /generate-rollout endpoint called")
    print(f"[DEBUG] Request: prompt='{request.prompt[:50]}...', n={request.n}, max_tokens={request.max_tokens}, temp={request.temperature}")

    try:
        # Create a temporary generator with custom settings if needed
        if request.max_tokens != 10000 or request.temperature != 0.7:
            print(f"[DEBUG] Creating temporary generator with max_tokens={request.max_tokens}, temp={request.temperature}")
            temp_generator = RolloutGenerator(
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            # Format the prompt using chat template
            formatted_prompt = temp_generator.apply_chat_template([{"role": "user", "content": request.prompt}])
            print(f"[DEBUG] Formatted prompt: {formatted_prompt[:100]}...")
            rollouts = temp_generator.generate(formatted_prompt, n=request.n, format_as_question=False)
        else:
            print(f"[DEBUG] Using default rollout_generator")
            # Format the prompt using chat template
            formatted_prompt = rollout_generator.apply_chat_template([{"role": "user", "content": request.prompt}])
            print(f"[DEBUG] Formatted prompt: {formatted_prompt[:100]}...")
            rollouts = rollout_generator.generate(formatted_prompt, n=request.n, format_as_question=False)

        print(f"[DEBUG] Successfully generated {len(rollouts)} rollouts")
        print(f"[DEBUG] First rollout preview: {rollouts[0][:200]}...")
        return GenerateRolloutResponse(rollouts=rollouts)

    except Exception as e:
        print(f"[ERROR] Failed to generate rollout: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate rollout: {str(e)}")


@app.post("/generate-interventions", response_model=InterventionResultResponse)
async def generate_interventions(request: GenerateInterventionsRequest):
    """
    Generate intervention candidates using VoiceInHeadStrategy.

    Returns all good interventions and indicates which one was selected.
    """
    print(f"\n[DEBUG] /generate-interventions endpoint called")
    print(f"[DEBUG] Request: rollout length={len(request.rollout)}, goal='{request.goal_intervention}', prompt='{request.original_prompt[:50]}...'")
    print(f"[DEBUG] Rollout preview: {request.rollout[:200]}...")

    try:
        # Apply voice-in-head intervention strategy
        print(f"[DEBUG] Calling intervention_inserter.apply()")
        # Note: intervened_text return value not used directly - we retrieve results from voice_strategy instance variables
        _, suggested_interventions = intervention_inserter.apply(
            rollout=request.rollout,
            intervention_text=request.goal_intervention,
            prompt=request.original_prompt
        )
        print(f"[DEBUG] Got {len(suggested_interventions)} suggested interventions")

        # Get the results from the strategy (it stores them as instance variables)
        clipped_text = voice_strategy.last_clipped_text
        good_interventions = voice_strategy.last_good_interventions
        selected_intervention = voice_strategy.last_selected_intervention
        selected_index = voice_strategy.last_selected_index

        print(f"[DEBUG] Returning {len(good_interventions)} good interventions, selected index {selected_index}")
        return InterventionResultResponse(
            clipped_text=clipped_text,
            generated_intervention=selected_intervention,
            all_good_interventions=good_interventions,
            selected_intervention_index=selected_index
        )

    except Exception as e:
        print(f"[ERROR] Failed to generate interventions: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate interventions: {str(e)}")


@app.post("/continue-from-intervention", response_model=ContinueFromInterventionResponse)
async def continue_from_intervention(request: ContinueFromInterventionRequest):
    """
    Continue generation from an intervention result.

    Takes the original prompt, clipped text, and generated intervention, then continues
    the completion with full context (original question + intervened text).
    """
    print(f"\n[DEBUG] /continue-from-intervention endpoint called")
    print(f"[DEBUG] Request: prompt='{request.original_prompt[:50]}...', clipped_text length={len(request.clipped_text)}, intervention length={len(request.generated_intervention)}")

    try:
        # Format the original prompt using chat template
        formatted_prompt = rollout_generator.apply_chat_template([
            {"role": "user", "content": request.original_prompt}
        ])

        # Create the intervened text for continuation (partial completion)
        intervened_text = f"<think>\n{request.clipped_text}{request.generated_intervention}"

        # Continue generation with full context (formatted_prompt + intervened_text)
        continuations = rollout_generator.continue_generation(
            formatted_prompt=formatted_prompt,
            partial_completion=intervened_text,
            n=1
        )
        continuation = continuations[0]

        # Full output includes the formatted prompt for accurate logprobs visualization
        full_output = formatted_prompt + intervened_text + continuation

        print(f"[DEBUG] Successfully generated continuation of length {len(continuation)}")
        return ContinueFromInterventionResponse(
            continuation=continuation,
            full_output=full_output
        )

    except Exception as e:
        print(f"[ERROR] Failed to continue generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to continue generation: {str(e)}")


@app.post("/get-logprobs", response_model=GetLogprobsResponse)
async def get_logprobs(request: GetLogprobsRequest):
    """
    Get logprobs for each token in the provided text.

    Returns token-level logprobs for visualization.
    """
    print(f"\n[DEBUG] /get-logprobs endpoint called")
    print(f"[DEBUG] Text length: {len(request.text)}")

    try:
        import requests as req

        payload = {
            "model": "Qwen/Qwen3-8b",
            "prompt": request.text,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "echo": True,
            "stream": False
        }

        response = req.post("http://localhost:8000/v1/completions", json=payload)
        response.raise_for_status()
        result = response.json()

        # Parse token logprobs
        tokens_with_logprobs = []
        prompt_logprobs = result['choices'][0].get('prompt_logprobs', [])

        # Skip first token (usually None)
        for token_dict in prompt_logprobs[1:]:
            if token_dict:
                for _, logprob_dict in token_dict.items():
                    if (len(token_dict) == 1) or (logprob_dict.get('rank') != 1):
                        tokens_with_logprobs.append(TokenLogprob(
                            token=logprob_dict['decoded_token'],
                            logprob=logprob_dict['logprob']
                        ))

        print(f"[DEBUG] Returning {len(tokens_with_logprobs)} tokens with logprobs")
        return GetLogprobsResponse(tokens_with_logprobs=tokens_with_logprobs)

    except Exception as e:
        print(f"[ERROR] Failed to get logprobs: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get logprobs: {str(e)}")


if __name__ == "__main__":
    print("Starting Voice-in-Head Pipeline API server...")
    print("Endpoints:")
    print("  - POST /generate-rollout")
    print("  - POST /generate-interventions")
    print("  - POST /continue-from-intervention")
    print("  - POST /get-logprobs")
    print("\nServer running on http://localhost:8002")

    uvicorn.run(app, host="0.0.0.0", port=8002)
