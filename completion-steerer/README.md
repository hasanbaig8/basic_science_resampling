# Completion Steerer

A React-based UI for steering LLM outputs by choosing from multiple completion candidates at each generation step. Built with Vite, React, TypeScript, and shadcn/ui.

## Features

- **Interactive Steering**: Choose from 10 completion candidates at each step
- **Chain-of-Thought Control**: See and select each thinking step including `<think>` tags
- **Step History**: Review and modify previous selections to explore alternative paths
- **Real-time Generation**: Connect directly to your local vLLM server
- **Beautiful UI**: Clean interface with shadcn/ui components and Tailwind CSS

## Prerequisites

- Node.js (v18+)
- A running vLLM server on `localhost:8000` with Qwen3-0.6b model

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Ensure your vLLM server is running:
```bash
# Example vLLM server command (run in a separate terminal)
vllm serve Qwen/Qwen3-0.6b --port 8000
```

## Usage

1. **Enter a Prompt**: Type your initial prompt in the text area
2. **Configure Parameters**:
   - Max Tokens: Number of tokens to generate per step (default: 50)
   - Temperature: Sampling temperature for diversity (default: 0.7)
3. **Start Generation**: Click "Start Generation" to get 10 completion options
4. **Select a Completion**: Choose your preferred continuation from the options
5. **Continue**: Click "Continue Generation" to extend the selected text with more options
6. **Review & Modify**: Expand step history to see alternative choices and change selections

## Key Features

### Step-by-Step Control
Each generation step gives you 10 different continuations to choose from. You can see the full text including chain-of-thought reasoning.

### Time Travel
Change any previous selection to explore different paths. When you change a selection, all subsequent steps are removed so you can continue from that new branch.

### Visual Feedback
- Each step is highlighted in the generated text
- Badges show which choice was selected
- Finish reasons indicate if generation is complete

## Tech Stack

- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Beautiful component library
- **Lucide React** - Icon library

## Project Structure

```
src/
├── components/
│   └── ui/          # shadcn/ui components
├── lib/
│   ├── utils.ts     # Utility functions
│   └── vllm-api.ts  # vLLM API integration
├── App.tsx          # Main application component
└── index.css        # Global styles
```

## API Integration

The app connects to a local vLLM server at `http://localhost:8000/v1/completions`. The API format matches OpenAI's completions endpoint:

```typescript
{
  model: "Qwen/Qwen3-0.6b",
  prompt: "<|im_start|>user\nYour prompt<|im_end|>\n<|im_start|>assistant\n",
  max_tokens: 50,
  temperature: 0.7,
  n: 10
}
```

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```
