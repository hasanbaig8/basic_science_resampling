# Quick Start Guide

## 1. Start your vLLM server

In a separate terminal, start your vLLM server:

```bash
vllm serve Qwen/Qwen3-0.6b --port 8000
```

Or if you have a specific command from your notebook setup, use that instead.

## 2. Start the UI

In the completion-steerer directory:

```bash
cd completion-steerer
npm run dev
```

This will start the development server, typically at `http://localhost:5173`

## 3. Use the UI

1. Open your browser to the URL shown in the terminal
2. Enter a prompt like: "What are the most common emotions that people feel? Think for a while"
3. Adjust max tokens (50) and temperature (0.7) if desired
4. Click "Start Generation"
5. Select one of the 10 completion options shown
6. Click "Continue Generation" to add more steps
7. Use the Step History panel to review and change previous selections

## Features

- **10 choices per step**: See all the chain-of-thought options
- **Time travel**: Click on any previous step to change your selection
- **Full visibility**: See the complete formatted prompt including chat template
- **Interactive steering**: Guide the model's reasoning by choosing paths

## Troubleshooting

### vLLM server not connecting
- Make sure vLLM is running on `localhost:8000`
- Check the browser console for CORS errors
- Verify the model name matches: `Qwen/Qwen3-0.6b`

### Build errors
- Node version warnings can be ignored if the dev server runs
- Make sure all dependencies are installed: `npm install`

## Architecture

The UI directly calls your vLLM server's completions API:
- No backend proxy needed
- Runs entirely in the browser
- State management with React hooks
- Real-time feedback with loading states
