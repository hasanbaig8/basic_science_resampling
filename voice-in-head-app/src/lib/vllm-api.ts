export interface CompletionChoice {
  text: string;
  index: number;
  finish_reason: string;
}

export interface CompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: CompletionChoice[];
}

export interface InterventionResult {
  clippedText: string;
  generatedIntervention: string;
  allGoodInterventions: string[];
  selectedInterventionIndex: number;
}

export interface VoiceInHeadResult extends InterventionResult {
  continuation: string;
  fullOutput: string;
}

export interface TokenLogprob {
  token: string;
  logprob: number;
}

// API base URL - uses Vite proxy (/api -> localhost:8002)
const API_BASE_URL = "/api";

/**
 * Generate rollout from a question (calls Python API)
 */
export async function generateRollout(
  prompt: string,
  n: number = 1,
  maxTokens: number = 1000,
  temperature: number = 0.7
): Promise<string[]> {
  const url = `${API_BASE_URL}/generate-rollout`;
  console.log(`[DEBUG] Calling ${url}`);

  const payload = {
    prompt: prompt,
    n: n,
    max_tokens: maxTokens,
    temperature: temperature,
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result.rollouts;
  } catch (error) {
    console.error("Error calling Python API for rollout generation:", error);
    throw error;
  }
}

export async function getCompletionsFromVLLM(
  prompt: string,
  maxTokens: number = 100,
  temperature: number = 0.7,
  n: number = 1
): Promise<CompletionChoice[]> {
  const url = "http://localhost:8000/v1/completions";

  const payload = {
    model: "Qwen/Qwen3-8b",
    prompt: prompt,
    max_tokens: maxTokens,
    temperature: temperature,
    stream: false,
    n: n,
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result: CompletionResponse = await response.json();
    return result.choices;
  } catch (error) {
    console.error("Error connecting to vLLM server:", error);
    throw error;
  }
}

export function formatPrompt(userPrompt: string): string {
  // Format prompt using Qwen3 chat template
  return `<|im_start|>user\n${userPrompt}<|im_end|>\n<|im_start|>assistant\n`;
}

/**
 * Generate intervention candidates without continuation (calls Python API)
 */
export async function generateInterventions(
  rollout: string,
  goalIntervention: string,
  originalPrompt: string
): Promise<InterventionResult> {
  const url = `${API_BASE_URL}/generate-interventions`;
  console.log(`[DEBUG] Calling ${url}`);

  const payload = {
    rollout: rollout,
    goal_intervention: goalIntervention,
    original_prompt: originalPrompt,
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return {
      clippedText: result.clipped_text,
      generatedIntervention: result.generated_intervention,
      allGoodInterventions: result.all_good_interventions,
      selectedInterventionIndex: result.selected_intervention_index,
    };
  } catch (error) {
    console.error("Error calling Python API for intervention generation:", error);
    throw error;
  }
}

/**
 * Continue generation from an intervention result (calls Python API)
 */
export async function continueFromIntervention(
  interventionResult: InterventionResult
): Promise<VoiceInHeadResult> {
  const url = `${API_BASE_URL}/continue-from-intervention`;
  console.log(`[DEBUG] Calling ${url}`);

  const payload = {
    clipped_text: interventionResult.clippedText,
    generated_intervention: interventionResult.generatedIntervention,
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return {
      ...interventionResult,
      continuation: result.continuation,
      fullOutput: result.full_output,
    };
  } catch (error) {
    console.error("Error calling Python API for continuation:", error);
    throw error;
  }
}

/**
 * Get logprobs for each token in text (calls Python API)
 */
export async function getLogprobs(text: string): Promise<TokenLogprob[]> {
  const url = `${API_BASE_URL}/get-logprobs`;
  console.log(`[DEBUG] Calling ${url}`);

  const payload = {
    text: text,
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result.tokens_with_logprobs;
  } catch (error) {
    console.error("Error calling Python API for logprobs:", error);
    throw error;
  }
}

/**
 * Apply voice-in-head intervention to a rollout (full pipeline)
 */
export async function applyVoiceInHeadIntervention(
  rollout: string,
  goalIntervention: string,
  originalPrompt: string
): Promise<VoiceInHeadResult> {
  const interventionResult = await generateInterventions(rollout, goalIntervention, originalPrompt);
  return await continueFromIntervention(interventionResult);
}
