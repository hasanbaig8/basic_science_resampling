import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { getCompletionsFromVLLM, formatPrompt, type CompletionChoice } from "@/lib/vllm-api";
import { Loader2, Sparkles, ChevronRight, Edit2, Save, X } from "lucide-react";

interface CompletionStep {
  text: string;
  choices: CompletionChoice[];
  selectedIndex: number;
}

function App() {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(50);
  const [temperature, setTemperature] = useState(0.7);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // History of completion steps
  const [completionSteps, setCompletionSteps] = useState<CompletionStep[]>([]);
  const [currentChoices, setCurrentChoices] = useState<CompletionChoice[]>([]);
  const [isSelectingFromChoices, setIsSelectingFromChoices] = useState(false);

  // Editing state
  const [isEditing, setIsEditing] = useState(false);
  const [editedText, setEditedText] = useState("");

  const getCurrentText = () => {
    if (completionSteps.length === 0) return "";
    return completionSteps.map(step => step.text).join("");
  };

  const generateCompletions = async (continueFromCurrent: boolean = false) => {
    setIsGenerating(true);
    setError(null);
    setIsSelectingFromChoices(false);

    try {
      let fullPrompt: string;

      if (continueFromCurrent && completionSteps.length > 0) {
        // Continue from current accumulated text
        const currentText = getCurrentText();
        fullPrompt = formatPrompt(prompt) + currentText;
      } else {
        // Start fresh
        fullPrompt = formatPrompt(prompt);
        setCompletionSteps([]);
      }

      const choices = await getCompletionsFromVLLM(
        fullPrompt,
        maxTokens,
        temperature,
        10
      );

      setCurrentChoices(choices);
      setIsSelectingFromChoices(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate completions");
    } finally {
      setIsGenerating(false);
    }
  };

  const selectChoice = (index: number) => {
    const selectedChoice = currentChoices[index];

    setCompletionSteps([
      ...completionSteps,
      {
        text: selectedChoice.text,
        choices: currentChoices,
        selectedIndex: index,
      },
    ]);

    setIsSelectingFromChoices(false);
    setCurrentChoices([]);
  };

  const removeStep = (stepIndex: number) => {
    setCompletionSteps(completionSteps.slice(0, stepIndex));
    setIsSelectingFromChoices(false);
    setCurrentChoices([]);
  };

  const changeSelection = (stepIndex: number, newChoiceIndex: number) => {
    const step = completionSteps[stepIndex];
    const newChoice = step.choices[newChoiceIndex];

    const newSteps = [...completionSteps];
    newSteps[stepIndex] = {
      ...step,
      text: newChoice.text,
      selectedIndex: newChoiceIndex,
    };

    // Remove all steps after this one
    setCompletionSteps(newSteps.slice(0, stepIndex + 1));
    setIsSelectingFromChoices(false);
    setCurrentChoices([]);
  };

  const reset = () => {
    setCompletionSteps([]);
    setCurrentChoices([]);
    setIsSelectingFromChoices(false);
    setError(null);
    setIsEditing(false);
    setEditedText("");
  };

  const startEditing = () => {
    setEditedText(getCurrentText());
    setIsEditing(true);
  };

  const cancelEditing = () => {
    setIsEditing(false);
    setEditedText("");
  };

  const saveEdits = () => {
    // Clear existing steps and create a single step with edited text
    setCompletionSteps([{
      text: editedText,
      choices: [],
      selectedIndex: 0
    }]);
    setIsEditing(false);
    setEditedText("");
  };

  const generateFromEdited = async () => {
    setIsGenerating(true);
    setError(null);
    setIsSelectingFromChoices(false);

    try {
      // Use edited text as the continuation point
      const fullPrompt = formatPrompt(prompt) + editedText;

      const choices = await getCompletionsFromVLLM(
        fullPrompt,
        maxTokens,
        temperature,
        10
      );

      setCurrentChoices(choices);
      setIsSelectingFromChoices(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate completions");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold flex items-center justify-center gap-2">
            <Sparkles className="w-8 h-8 text-blue-500" />
            Completion Steerer
          </h1>
          <p className="text-muted-foreground">
            Steer LLM outputs by choosing from multiple completions at each step
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Input & Controls */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Prompt</CardTitle>
                <CardDescription>Enter your initial prompt</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="What are the most common emotions that people feel? Think for a while"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={4}
                  className="resize-none"
                />

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Max Tokens</label>
                    <input
                      type="number"
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(Number(e.target.value))}
                      className="w-full px-3 py-2 border rounded-md"
                      min={10}
                      max={500}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Temperature</label>
                    <input
                      type="number"
                      value={temperature}
                      onChange={(e) => setTemperature(Number(e.target.value))}
                      className="w-full px-3 py-2 border rounded-md"
                      min={0}
                      max={2}
                      step={0.1}
                    />
                  </div>
                </div>

                <div className="flex gap-2">
                  <Button
                    onClick={() => generateCompletions(false)}
                    disabled={!prompt || isGenerating}
                    className="flex-1"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      "Start Generation"
                    )}
                  </Button>
                  {completionSteps.length > 0 && (
                    <Button onClick={reset} variant="outline">
                      Reset
                    </Button>
                  )}
                </div>

                {error && (
                  <div className="p-3 bg-destructive/10 text-destructive rounded-md text-sm">
                    {error}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Accumulated Output */}
            {completionSteps.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Generated Text</CardTitle>
                  <CardDescription>
                    {isEditing
                      ? "Editing mode - make your changes below"
                      : `${completionSteps.length} step${completionSteps.length !== 1 ? "s" : ""} completed`
                    }
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {isEditing ? (
                      // Edit mode: show textarea
                      <div className="space-y-2">
                        <Textarea
                          value={editedText}
                          onChange={(e) => setEditedText(e.target.value)}
                          rows={15}
                          className="font-mono text-sm"
                          placeholder="Edit the assistant response here..."
                        />
                        <div className="flex gap-2">
                          <Button
                            onClick={saveEdits}
                            size="sm"
                            className="flex-1"
                          >
                            <Save className="w-4 h-4 mr-2" />
                            Save Changes
                          </Button>
                          <Button
                            onClick={cancelEditing}
                            variant="outline"
                            size="sm"
                            className="flex-1"
                          >
                            <X className="w-4 h-4 mr-2" />
                            Cancel
                          </Button>
                        </div>
                        <Button
                          onClick={generateFromEdited}
                          disabled={isGenerating || !editedText.trim()}
                          size="sm"
                          className="w-full"
                        >
                          {isGenerating ? (
                            <>
                              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                              Generating...
                            </>
                          ) : (
                            <>
                              <ChevronRight className="w-4 h-4 mr-2" />
                              Generate from Edited Text
                            </>
                          )}
                        </Button>
                      </div>
                    ) : (
                      // View mode: show completed text
                      <>
                        <div className="p-4 bg-muted rounded-lg max-h-96 overflow-y-auto">
                          <p className="text-sm text-muted-foreground mb-2 font-mono">
                            {formatPrompt(prompt)}
                          </p>
                          {completionSteps.map((step, idx) => (
                            <div key={idx} className="relative group">
                              <span className="text-sm font-mono bg-blue-100 dark:bg-blue-900/30 px-1 rounded">
                                {step.text}
                              </span>
                              <Badge variant="outline" className="ml-2 text-xs">
                                Step {idx + 1}
                              </Badge>
                            </div>
                          ))}
                        </div>

                        <div className="flex gap-2">
                          <Button
                            onClick={() => generateCompletions(true)}
                            disabled={isGenerating}
                            size="sm"
                            className="flex-1"
                          >
                            <ChevronRight className="w-4 h-4 mr-2" />
                            Continue Generation
                          </Button>
                          <Button
                            onClick={startEditing}
                            variant="outline"
                            size="sm"
                            className="flex-1"
                          >
                            <Edit2 className="w-4 h-4 mr-2" />
                            Edit Text
                          </Button>
                        </div>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column: Choices */}
          <div className="space-y-4">
            {isSelectingFromChoices && currentChoices.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Select a Completion</CardTitle>
                  <CardDescription>
                    Choose from {currentChoices.length} possible continuations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {currentChoices.map((choice, idx) => (
                      <button
                        key={idx}
                        onClick={() => selectChoice(idx)}
                        className="w-full text-left p-4 border rounded-lg hover:bg-accent hover:border-blue-500 transition-all group"
                      >
                        <div className="flex items-start justify-between gap-2 mb-2">
                          <Badge variant="outline" className="text-xs">
                            Choice {idx + 1}
                          </Badge>
                          <Badge
                            variant={
                              choice.finish_reason === "stop"
                                ? "default"
                                : "secondary"
                            }
                            className="text-xs"
                          >
                            {choice.finish_reason}
                          </Badge>
                        </div>
                        <p className="text-sm font-mono leading-relaxed whitespace-pre-wrap break-words">
                          {choice.text}
                        </p>
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Step History with Alternative Choices */}
            {completionSteps.length > 0 && !isSelectingFromChoices && (
              <Card>
                <CardHeader>
                  <CardTitle>Step History</CardTitle>
                  <CardDescription>
                    View and modify previous selections
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 max-h-[600px] overflow-y-auto">
                    {completionSteps.map((step, stepIdx) => (
                      <div key={stepIdx} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-semibold">
                            Step {stepIdx + 1}
                          </h4>
                          <Button
                            onClick={() => removeStep(stepIdx)}
                            variant="ghost"
                            size="sm"
                          >
                            Remove
                          </Button>
                        </div>
                        <details className="border rounded-lg">
                          <summary className="p-3 cursor-pointer hover:bg-accent">
                            <div className="flex items-start gap-2">
                              <Badge variant="outline" className="text-xs">
                                Selected: {step.selectedIndex + 1}
                              </Badge>
                              <p className="text-sm font-mono flex-1 line-clamp-2">
                                {step.text}
                              </p>
                            </div>
                          </summary>
                          <div className="p-3 pt-0 space-y-2 border-t">
                            <p className="text-xs text-muted-foreground mb-2">
                              Alternative choices:
                            </p>
                            {step.choices.map((choice, choiceIdx) => (
                              <button
                                key={choiceIdx}
                                onClick={() =>
                                  changeSelection(stepIdx, choiceIdx)
                                }
                                disabled={choiceIdx === step.selectedIndex}
                                className={`w-full text-left p-2 text-xs rounded border ${
                                  choiceIdx === step.selectedIndex
                                    ? "bg-blue-100 dark:bg-blue-900/30 border-blue-500"
                                    : "hover:bg-accent"
                                } transition-all disabled:cursor-not-allowed`}
                              >
                                <Badge variant="outline" className="text-xs mb-1">
                                  Choice {choiceIdx + 1}
                                </Badge>
                                <p className="font-mono line-clamp-2">
                                  {choice.text}
                                </p>
                              </button>
                            ))}
                          </div>
                        </details>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {!isSelectingFromChoices && completionSteps.length === 0 && (
              <Card className="border-dashed">
                <CardContent className="flex items-center justify-center h-64">
                  <p className="text-muted-foreground text-center">
                    Enter a prompt and click "Start Generation" to begin
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
