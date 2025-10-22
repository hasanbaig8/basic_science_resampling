import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  generateRollout,
  generateInterventions,
  continueFromIntervention,
  applyVoiceInHeadIntervention,
  getLogprobs,
  type InterventionResult,
  type VoiceInHeadResult,
  type TokenLogprob,
} from "@/lib/vllm-api";
import { Loader2, Sparkles, Brain, ChevronDown, ChevronRight, Play } from "lucide-react";

// Helper function to get color based on logprob using continuous spectrum
function getLogprobColor(logprob: number): string {
  // Logprobs typically range from -10 to 0
  // Higher (closer to 0) = more confident = greener
  // Lower (more negative) = less confident = redder
  const normalized = Math.max(0, Math.min(1, (logprob + 10) / 10));

  // Create continuous RGB gradient from red (low confidence) to green (high confidence)
  // Red: rgb(239, 68, 68) -> Yellow: rgb(234, 179, 8) -> Green: rgb(34, 197, 94)
  let r, g, b;

  if (normalized < 0.5) {
    // Red to Yellow
    const t = normalized * 2; // 0 to 1
    r = Math.round(239 + (234 - 239) * t);
    g = Math.round(68 + (179 - 68) * t);
    b = Math.round(68 + (8 - 68) * t);
  } else {
    // Yellow to Green
    const t = (normalized - 0.5) * 2; // 0 to 1
    r = Math.round(234 + (34 - 234) * t);
    g = Math.round(179 + (197 - 179) * t);
    b = Math.round(8 + (94 - 8) * t);
  }

  return `rgb(${r}, ${g}, ${b})`;
}

function App() {
  const [prompt, setPrompt] = useState("What are some fun things to do in London?");
  const [goalIntervention, setGoalIntervention] = useState("Go for a day trip to Croydon.");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingLogprobs, setIsLoadingLogprobs] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generated content
  const [initialRollout, setInitialRollout] = useState<string>("");
  const [interventionResult, setInterventionResult] = useState<InterventionResult | null>(null);
  const [result, setResult] = useState<VoiceInHeadResult | null>(null);
  const [isAlternativesOpen, setIsAlternativesOpen] = useState(false);
  const [tokensWithLogprobs, setTokensWithLogprobs] = useState<TokenLogprob[] | null>(null);

  const generateInitialRollout = async () => {
    setIsGenerating(true);
    setError(null);
    setInitialRollout("");
    setInterventionResult(null);
    setResult(null);

    try {
      const rollouts = await generateRollout(prompt, 1, 1000, 0.7);
      setInitialRollout(rollouts[0]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate initial rollout");
    } finally {
      setIsGenerating(false);
    }
  };

  const generateInterventionOnly = async () => {
    if (!initialRollout) {
      setError("Please generate an initial rollout first");
      return;
    }

    setIsGenerating(true);
    setError(null);
    setResult(null);

    try {
      const intervention = await generateInterventions(
        initialRollout,
        goalIntervention,
        prompt
      );
      setInterventionResult(intervention);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate intervention");
    } finally {
      setIsGenerating(false);
    }
  };

  const loadLogprobs = async () => {
    if (!result) {
      setError("No result to load logprobs for");
      return;
    }

    setIsLoadingLogprobs(true);
    setError(null);

    try {
      const logprobs = await getLogprobs(result.fullOutput);
      setTokensWithLogprobs(logprobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load logprobs");
    } finally {
      setIsLoadingLogprobs(false);
    }
  };

  const generateContinuation = async () => {
    if (!interventionResult) {
      setError("Please generate intervention first");
      return;
    }

    setIsGenerating(true);
    setError(null);
    setTokensWithLogprobs(null);

    try {
      const fullResult = await continueFromIntervention(interventionResult, prompt);
      setResult(fullResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate continuation");
    } finally {
      setIsGenerating(false);
    }
  };

  const runFullPipeline = async () => {
    setIsGenerating(true);
    setError(null);
    setInitialRollout("");
    setInterventionResult(null);
    setResult(null);
    setTokensWithLogprobs(null);

    try {
      // Step 1: Generate initial rollout
      const rollouts = await generateRollout(prompt, 1, 1000, 0.7);
      const rollout = rollouts[0];
      setInitialRollout(rollout);

      // Step 2: Apply intervention (full pipeline)
      const fullResult = await applyVoiceInHeadIntervention(
        rollout,
        goalIntervention,
        prompt
      );
      setInterventionResult(fullResult); // Store intervention part too
      setResult(fullResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run pipeline");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Brain className="w-10 h-10 text-blue-600" />
            <h1 className="text-4xl font-bold text-slate-900">Voice-in-Head Intervention</h1>
          </div>
          <p className="text-slate-600">
            Interrupt LLM reasoning at random early positions (15-35%) and steer the output
          </p>
        </div>

        {error && (
          <Card className="border-red-200 bg-red-50">
            <CardContent className="pt-6">
              <p className="text-red-800">{error}</p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>Set your prompt and intervention goal</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Prompt</label>
              <Textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt..."
                className="min-h-[100px]"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Goal Intervention</label>
              <Input
                value={goalIntervention}
                onChange={(e) => setGoalIntervention(e.target.value)}
                placeholder="What should the model steer towards?"
              />
            </div>
            <div className="space-y-3">
              <div className="flex gap-2">
                <Button
                  onClick={runFullPipeline}
                  disabled={isGenerating || !prompt || !goalIntervention}
                  className="flex-1"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-4 w-4" />
                      Run Full Pipeline
                    </>
                  )}
                </Button>
              </div>
              <div className="flex gap-2">
                <Button
                  onClick={generateInitialRollout}
                  disabled={isGenerating || !prompt}
                  variant="outline"
                  className="flex-1"
                >
                  1. Generate Rollout
                </Button>
                <Button
                  onClick={generateInterventionOnly}
                  disabled={isGenerating || !initialRollout || !goalIntervention}
                  variant="outline"
                  className="flex-1"
                >
                  2. Generate Interventions
                </Button>
                <Button
                  onClick={generateContinuation}
                  disabled={isGenerating || !interventionResult}
                  variant="outline"
                  className="flex-1"
                >
                  3. Continue
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {initialRollout && (
          <Card>
            <CardHeader>
              <CardTitle>Initial Rollout</CardTitle>
              <CardDescription>Generated reasoning from the model</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-100 p-4 rounded-lg font-mono text-sm whitespace-pre-wrap max-h-96 overflow-y-auto">
                {initialRollout}
              </div>
            </CardContent>
          </Card>
        )}

        {interventionResult && !result && (
          <Card>
            <CardHeader>
              <CardTitle>Generated Interventions</CardTitle>
              <CardDescription>
                View intervention candidates before continuing generation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-semibold text-blue-600 mb-2">Clipped Original Text</h3>
                <div className="bg-blue-50 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {interventionResult.clippedText}
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-green-600">Selected Intervention</h3>
                  {interventionResult.allGoodInterventions.length > 1 && (
                    <Collapsible open={isAlternativesOpen} onOpenChange={setIsAlternativesOpen}>
                      <CollapsibleTrigger asChild>
                        <Button variant="outline" size="sm" className="gap-2">
                          {isAlternativesOpen ? (
                            <>
                              <ChevronDown className="h-4 w-4" />
                              Hide Alternatives ({interventionResult.allGoodInterventions.length - 1})
                            </>
                          ) : (
                            <>
                              <ChevronRight className="h-4 w-4" />
                              Show Alternatives ({interventionResult.allGoodInterventions.length - 1})
                            </>
                          )}
                        </Button>
                      </CollapsibleTrigger>
                    </Collapsible>
                  )}
                </div>
                <div className="bg-green-50 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {interventionResult.generatedIntervention}
                </div>
                {interventionResult.allGoodInterventions.length > 1 && (
                  <Collapsible open={isAlternativesOpen} onOpenChange={setIsAlternativesOpen}>
                    <CollapsibleContent className="mt-3">
                      <div className="space-y-2">
                        <p className="text-xs text-slate-600 font-semibold uppercase tracking-wide">
                          Alternative Interventions ({interventionResult.allGoodInterventions.length} total)
                        </p>
                        <div className="space-y-2 max-h-96 overflow-y-auto">
                          {interventionResult.allGoodInterventions.map((intervention, idx) => (
                            <div
                              key={idx}
                              className={`p-3 rounded text-sm font-mono whitespace-pre-wrap border ${
                                idx === interventionResult.selectedInterventionIndex
                                  ? "bg-green-100 border-green-400 border-2"
                                  : "bg-slate-50 border-slate-200"
                              }`}
                            >
                              <div className="flex items-start gap-2">
                                <span className="text-xs font-bold text-slate-500 min-w-[3rem]">
                                  #{idx + 1}
                                  {idx === interventionResult.selectedInterventionIndex && (
                                    <Badge className="ml-2 bg-green-600 text-xs">Selected</Badge>
                                  )}
                                </span>
                                <span className="flex-1">{intervention}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                )}
              </div>
              <div className="pt-4 border-t">
                <Button
                  onClick={generateContinuation}
                  disabled={isGenerating}
                  className="w-full"
                >
                  {isGenerating ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating Continuation...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Generate Continuation
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Intervention Result</CardTitle>
              <CardDescription>
                {tokensWithLogprobs
                  ? "Color-coded by token logprobs (green = high confidence, red = low confidence)"
                  : "Color-coded output showing the intervention process"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {tokensWithLogprobs ? (
                <>
                  <div className="flex gap-4 text-sm flex-wrap">
                    <Badge className="bg-green-200 text-black">High Confidence</Badge>
                    <Badge className="bg-yellow-100 text-black">Medium Confidence</Badge>
                    <Badge className="bg-red-100 text-black">Low Confidence</Badge>
                  </div>
                  <div className="bg-slate-50 p-4 rounded-lg font-mono text-sm leading-relaxed">
                    {tokensWithLogprobs.map((item, idx) => (
                      <span
                        key={idx}
                        className="px-0.5"
                        style={{ backgroundColor: getLogprobColor(item.logprob) }}
                        title={`Token: "${item.token}" | Logprob: ${item.logprob.toFixed(3)}`}
                      >
                        {item.token}
                      </span>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <div className="flex gap-4 text-sm">
                    <Badge className="bg-blue-500">Clipped Original</Badge>
                    <Badge className="bg-green-500">Generated Intervention</Badge>
                    <Badge className="bg-yellow-500 text-black">Continuation</Badge>
                  </div>
                  <div className="bg-slate-50 p-4 rounded-lg font-mono text-sm whitespace-pre-wrap leading-relaxed">
                    <span className="bg-blue-200 px-1 py-0.5 rounded">{result.clippedText}</span>
                    <span className="bg-green-200 px-1 py-0.5 rounded">
                      {result.generatedIntervention}
                    </span>
                    <span className="bg-yellow-200 px-1 py-0.5 rounded">{result.continuation}</span>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        )}

        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Component Breakdown</CardTitle>
              <CardDescription>
                {tokensWithLogprobs
                  ? "Color-coded by token logprobs (green = high confidence, red = low confidence)"
                  : "View each component separately"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {!tokensWithLogprobs && (
                <div className="mb-4">
                  <Button
                    onClick={loadLogprobs}
                    disabled={isLoadingLogprobs}
                    variant="outline"
                    className="w-full"
                  >
                    {isLoadingLogprobs ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Loading Logprobs Visualization...
                      </>
                    ) : (
                      <>
                        <Sparkles className="mr-2 h-4 w-4" />
                        Load Logprobs Visualization
                      </>
                    )}
                  </Button>
                  <p className="text-xs text-slate-500 mt-2 text-center">
                    Optional: Load token-level confidence colors (requires vLLM processing)
                  </p>
                </div>
              )}
              {tokensWithLogprobs ? (
                <>
                  <div className="flex gap-4 text-sm flex-wrap mb-4">
                    <Badge className="bg-green-200 text-black">High Confidence</Badge>
                    <Badge className="bg-yellow-100 text-black">Medium Confidence</Badge>
                    <Badge className="bg-red-100 text-black">Low Confidence</Badge>
                  </div>
                  <div>
                    <h3 className="font-semibold text-blue-600 mb-2">Clipped Original Text</h3>
                    <div className="bg-slate-50 p-3 rounded text-sm font-mono leading-relaxed max-h-48 overflow-y-auto">
                      {(() => {
                        let charCount = 0;
                        return tokensWithLogprobs.filter((item) => {
                          const prevCount = charCount;
                          charCount += item.token.length;
                          return prevCount < result.clippedText.length;
                        }).map((item, idx) => (
                          <span
                            key={idx}
                            className="px-0.5"
                            style={{ backgroundColor: getLogprobColor(item.logprob) }}
                            title={`Token: "${item.token}" | Logprob: ${item.logprob.toFixed(3)}`}
                          >
                            {item.token}
                          </span>
                        ));
                      })()}
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-green-600">Generated Intervention</h3>
                      {result.allGoodInterventions.length > 1 && (
                        <Collapsible open={isAlternativesOpen} onOpenChange={setIsAlternativesOpen}>
                          <CollapsibleTrigger asChild>
                            <Button variant="outline" size="sm" className="gap-2">
                              {isAlternativesOpen ? (
                                <>
                                  <ChevronDown className="h-4 w-4" />
                                  Hide Alternatives ({result.allGoodInterventions.length - 1})
                                </>
                              ) : (
                                <>
                                  <ChevronRight className="h-4 w-4" />
                                  Show Alternatives ({result.allGoodInterventions.length - 1})
                                </>
                              )}
                            </Button>
                          </CollapsibleTrigger>
                        </Collapsible>
                      )}
                    </div>
                    <div className="bg-slate-50 p-3 rounded text-sm font-mono leading-relaxed max-h-48 overflow-y-auto">
                      {(() => {
                        let charCount = 0;
                        const startChar = result.clippedText.length;
                        const endChar = startChar + result.generatedIntervention.length;
                        return tokensWithLogprobs.filter((item) => {
                          const prevCount = charCount;
                          charCount += item.token.length;
                          return prevCount >= startChar && prevCount < endChar;
                        }).map((item, idx) => (
                          <span
                            key={idx}
                            className="px-0.5"
                            style={{ backgroundColor: getLogprobColor(item.logprob) }}
                            title={`Token: "${item.token}" | Logprob: ${item.logprob.toFixed(3)}`}
                          >
                            {item.token}
                          </span>
                        ));
                      })()}
                    </div>
                    {result.allGoodInterventions.length > 1 && (
                      <Collapsible open={isAlternativesOpen} onOpenChange={setIsAlternativesOpen}>
                        <CollapsibleContent className="mt-3">
                          <div className="space-y-2">
                            <p className="text-xs text-slate-600 font-semibold uppercase tracking-wide">
                              Alternative Interventions ({result.allGoodInterventions.length} total)
                            </p>
                            <div className="space-y-2 max-h-96 overflow-y-auto">
                              {result.allGoodInterventions.map((intervention, idx) => (
                                <div
                                  key={idx}
                                  className={`p-3 rounded text-sm font-mono whitespace-pre-wrap border ${
                                    idx === result.selectedInterventionIndex
                                      ? "bg-green-100 border-green-400 border-2"
                                      : "bg-slate-50 border-slate-200"
                                  }`}
                                >
                                  <div className="flex items-start gap-2">
                                    <span className="text-xs font-bold text-slate-500 min-w-[3rem]">
                                      #{idx + 1}
                                      {idx === result.selectedInterventionIndex && (
                                        <Badge className="ml-2 bg-green-600 text-xs">Selected</Badge>
                                      )}
                                    </span>
                                    <span className="flex-1">{intervention}</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </CollapsibleContent>
                      </Collapsible>
                    )}
                  </div>
                  <div>
                    <h3 className="font-semibold text-purple-600 mb-2">Continuation</h3>
                    <div className="bg-slate-50 p-3 rounded text-sm font-mono leading-relaxed max-h-96 overflow-y-auto">
                      {(() => {
                        let charCount = 0;
                        const startChar = result.clippedText.length + result.generatedIntervention.length;
                        return tokensWithLogprobs.filter((item) => {
                          const prevCount = charCount;
                          charCount += item.token.length;
                          return prevCount >= startChar;
                        }).map((item, idx) => (
                          <span
                            key={idx}
                            className="px-0.5"
                            style={{ backgroundColor: getLogprobColor(item.logprob) }}
                            title={`Token: "${item.token}" | Logprob: ${item.logprob.toFixed(3)}`}
                          >
                            {item.token}
                          </span>
                        ));
                      })()}
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <h3 className="font-semibold text-blue-600 mb-2">Clipped Original Text</h3>
                    <div className="bg-blue-50 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                      {result.clippedText}
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-green-600">Generated Intervention</h3>
                      {result.allGoodInterventions.length > 1 && (
                        <Collapsible open={isAlternativesOpen} onOpenChange={setIsAlternativesOpen}>
                          <CollapsibleTrigger asChild>
                            <Button variant="outline" size="sm" className="gap-2">
                              {isAlternativesOpen ? (
                                <>
                                  <ChevronDown className="h-4 w-4" />
                                  Hide Alternatives ({result.allGoodInterventions.length - 1})
                                </>
                              ) : (
                                <>
                                  <ChevronRight className="h-4 w-4" />
                                  Show Alternatives ({result.allGoodInterventions.length - 1})
                                </>
                              )}
                            </Button>
                          </CollapsibleTrigger>
                        </Collapsible>
                      )}
                    </div>
                    <div className="bg-green-50 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                      {result.generatedIntervention}
                    </div>
                    {result.allGoodInterventions.length > 1 && (
                      <Collapsible open={isAlternativesOpen} onOpenChange={setIsAlternativesOpen}>
                        <CollapsibleContent className="mt-3">
                          <div className="space-y-2">
                            <p className="text-xs text-slate-600 font-semibold uppercase tracking-wide">
                              Alternative Interventions ({result.allGoodInterventions.length} total)
                            </p>
                            <div className="space-y-2 max-h-96 overflow-y-auto">
                              {result.allGoodInterventions.map((intervention, idx) => (
                                <div
                                  key={idx}
                                  className={`p-3 rounded text-sm font-mono whitespace-pre-wrap border ${
                                    idx === result.selectedInterventionIndex
                                      ? "bg-green-100 border-green-400 border-2"
                                      : "bg-slate-50 border-slate-200"
                                  }`}
                                >
                                  <div className="flex items-start gap-2">
                                    <span className="text-xs font-bold text-slate-500 min-w-[3rem]">
                                      #{idx + 1}
                                      {idx === result.selectedInterventionIndex && (
                                        <Badge className="ml-2 bg-green-600 text-xs">Selected</Badge>
                                      )}
                                    </span>
                                    <span className="flex-1">{intervention}</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </CollapsibleContent>
                      </Collapsible>
                    )}
                  </div>
                  <div>
                    <h3 className="font-semibold text-purple-600 mb-2">Continuation</h3>
                    <div className="bg-purple-50 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-96 overflow-y-auto">
                      {result.continuation}
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

export default App;
