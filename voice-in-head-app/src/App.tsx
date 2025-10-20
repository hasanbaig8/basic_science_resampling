import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  getCompletionsFromVLLM,
  formatPrompt,
  applyVoiceInHeadIntervention,
  type VoiceInHeadResult,
} from "@/lib/vllm-api";
import { Loader2, Sparkles, Brain } from "lucide-react";

function App() {
  const [prompt, setPrompt] = useState("What are some fun things to do in London?");
  const [goalIntervention, setGoalIntervention] = useState("Go for a day trip to Croydon.");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generated content
  const [initialRollout, setInitialRollout] = useState<string>("");
  const [result, setResult] = useState<VoiceInHeadResult | null>(null);

  const generateInitialRollout = async () => {
    setIsGenerating(true);
    setError(null);
    setInitialRollout("");
    setResult(null);

    try {
      const formattedPrompt = formatPrompt(prompt);
      const choices = await getCompletionsFromVLLM(formattedPrompt, 1000, 0.7, 1);
      const rollout = choices[0].text;
      setInitialRollout(rollout);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate initial rollout");
    } finally {
      setIsGenerating(false);
    }
  };

  const applyIntervention = async () => {
    if (!initialRollout) {
      setError("Please generate an initial rollout first");
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const interventionResult = await applyVoiceInHeadIntervention(
        initialRollout,
        goalIntervention,
        prompt
      );
      setResult(interventionResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to apply intervention");
    } finally {
      setIsGenerating(false);
    }
  };

  const runFullPipeline = async () => {
    setIsGenerating(true);
    setError(null);
    setInitialRollout("");
    setResult(null);

    try {
      // Step 1: Generate initial rollout
      const formattedPrompt = formatPrompt(prompt);
      const choices = await getCompletionsFromVLLM(formattedPrompt, 1000, 0.7, 1);
      const rollout = choices[0].text;
      setInitialRollout(rollout);

      // Step 2: Apply intervention
      const interventionResult = await applyVoiceInHeadIntervention(
        rollout,
        goalIntervention,
        prompt
      );
      setResult(interventionResult);
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
              <Button
                onClick={generateInitialRollout}
                disabled={isGenerating || !prompt}
                variant="outline"
              >
                1. Generate Rollout
              </Button>
              <Button
                onClick={applyIntervention}
                disabled={isGenerating || !initialRollout || !goalIntervention}
                variant="outline"
              >
                2. Apply Intervention
              </Button>
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

        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Intervention Result</CardTitle>
              <CardDescription>
                Color-coded output showing the intervention process
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
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
            </CardContent>
          </Card>
        )}

        {result && (
          <Card>
            <CardHeader>
              <CardTitle>Component Breakdown</CardTitle>
              <CardDescription>View each component separately</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-semibold text-blue-600 mb-2">Clipped Original Text</h3>
                <div className="bg-blue-50 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {result.clippedText}
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-green-600 mb-2">Generated Intervention</h3>
                <div className="bg-green-50 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {result.generatedIntervention}
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-yellow-600 mb-2">Continuation</h3>
                <div className="bg-yellow-50 p-3 rounded text-sm font-mono whitespace-pre-wrap max-h-96 overflow-y-auto">
                  {result.continuation}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

export default App;
