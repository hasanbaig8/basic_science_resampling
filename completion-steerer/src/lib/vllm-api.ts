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

export async function getCompletionsFromVLLM(
  prompt: string,
  maxTokens: number = 100,
  temperature: number = 0.7,
  n: number = 10
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

export function formatPrompt(prompt: string): string {
  // Format prompt using Qwen3 chat template
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
}
