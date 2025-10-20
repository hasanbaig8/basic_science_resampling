# Voice-in-Head Intervention App

Interactive React application for testing the Voice-in-Head intervention strategy on LLM reasoning.

## Features

- **Auto-generate initial rollouts** from custom prompts
- **Apply voice-in-head interventions** at random early positions (15-35%)
- **Color-coded display** showing:
  - 🔵 Blue: Clipped original text
  - 🟢 Green: Generated intervention (model's steering attempt)
  - 🟡 Yellow: Final continuation
- **Component breakdown** view for detailed analysis
- **Full pipeline mode** - one-click generation and intervention

## Tech Stack

- React 19 + TypeScript
- Vite for fast development
- Tailwind CSS for styling
- shadcn/ui components
- vLLM API integration

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start vLLM server** (required):
   ```bash
   vllm serve Qwen/Qwen3-8b --port 8000
   ```

3. **Run development server:**
   ```bash
   npm run dev
   ```

4. **Open in browser:**
   ```
   http://localhost:5174
   ```

## Usage

1. Enter your prompt (e.g., "What are some fun things to do in London?")
2. Set your goal intervention (e.g., "Go for a day trip to Croydon.")
3. Click "Run Full Pipeline" to:
   - Generate an initial rollout
   - Apply voice-in-head intervention
   - Continue generation with steering
4. View the color-coded result showing the three components

## Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Lint TypeScript code
- `npm run preview` - Preview production build

## How It Works

The voice-in-head strategy:
1. Clips the original rollout at a random position between 15-35%
2. Creates an intervention prompt asking the model to steer towards the goal
3. Generates 30 candidate interventions
4. Filters out bad interventions (containing "user")
5. Randomly selects a good intervention
6. Continues generation from the intervened text

This simulates a "voice in head" that interrupts early in the reasoning process and steers the model towards a specific goal.
