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

export interface VoiceInHeadResult {
  clippedText: string;
  generatedIntervention: string;
  continuation: string;
  fullOutput: string;
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
 * Apply voice-in-head intervention to a rollout
 */
export async function applyVoiceInHeadIntervention(
  rollout: string,
  goalIntervention: string,
  originalPrompt: string
): Promise<VoiceInHeadResult> {
  // Extract <think> content from rollout (only need opening tag)
  const thinkMatch = rollout.match(/<think>(.*)/s);
  if (!thinkMatch) {
    throw new Error("No <think> tag found in rollout");
  }
  const textToClip = thinkMatch[1];

  // Choose a random position between 15% and 35%
  const positionPct = Math.random() * 0.2 + 0.15; // 0.15 to 0.35
  const clipPosition = Math.floor(textToClip.length * positionPct);
  const clippedRaw = textToClip.slice(0, clipPosition);

  // Clip at word boundary
  const spaceSplit = clippedRaw.split(" ");
  const clippedText = spaceSplit.slice(0, -1).join(" ");
  const starter = spaceSplit.slice(-6, -1).join(" ");

  // Create voice-in-head intervention prompt
  const interventionPrompt =
    `<think>\n${clippedText}</think><|im_end|>\n<|im_start|>user\n` +
    `I interrupted you. Continue, and steer the response towards ${goalIntervention} within the next 3 sentences<|im_end|>\n` +
    `<|im_start|>assistant\n${starter}`;

  const templatedPrompt = formatPrompt(originalPrompt) + interventionPrompt;

  // Generate intervention suggestions (30 candidates)
  const suggestedInterventions = await getCompletionsFromVLLM(templatedPrompt, 100, 0.7, 30);

  // Filter out bad interventions (those containing "user")
  const goodInterventions = suggestedInterventions.filter(
    (intervention) => !intervention.text.includes("user")
  );

  if (goodInterventions.length === 0) {
    console.warn("No good interventions found, using first suggestion");
  }

  // Select a random good intervention
  const selectedIntervention = goodInterventions.length > 0
    ? goodInterventions[Math.floor(Math.random() * goodInterventions.length)]
    : suggestedInterventions[0];

  let generatedIntervention = selectedIntervention.text;

  // Clean up intervention if it contains </think>
  if (generatedIntervention.includes("</think>")) {
    generatedIntervention = generatedIntervention.split("</think>")[0];
  }

  // Create the intervened text for continuation
  const intervenedText = `<think>\n${clippedText}${generatedIntervention}`;

  // Continue generation from the intervened text
  const continuationChoices = await getCompletionsFromVLLM(intervenedText, 10000, 0.7, 1);
  const continuation = continuationChoices[0].text;

  return {
    clippedText,
    generatedIntervention,
    continuation,
    fullOutput: intervenedText + continuation,
  };
}
