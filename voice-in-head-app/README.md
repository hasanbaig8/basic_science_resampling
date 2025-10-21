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

**You need 3 terminals running simultaneously:**

### Terminal 1: vLLM Server
```bash
vllm serve Qwen/Qwen3-8b --port 8000
```
This is the LLM inference server (required for generation).

### Terminal 2: FastAPI Pipeline Server
```bash
# From the project root
source .venv/bin/activate
python api_server.py
```
This runs on port 8002 and exposes the Python pipeline as REST APIs.

### Terminal 3: React Development Server
```bash
# Install dependencies (first time only)
npm install

# Run development server
npm run dev
```
This runs on port 5174. Open browser to **http://localhost:5174**

**Note:** The React app proxies `/api/*` requests to the FastAPI server internally, so all 3 servers must be running.

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

The voice-in-head strategy (implemented in Python `pipeline/voice_in_head_strategy.py`):
1. Clips the original rollout at a random position between 15-35%
2. Creates an intervention prompt asking the model to steer towards the goal
3. Generates 30 candidate interventions
4. Filters out bad interventions (containing "user" or "steer")
5. Uses sentence embeddings to select the intervention most similar to the goal
6. Continues generation from the intervened text

This simulates a "voice in head" that interrupts early in the reasoning process and steers the model towards a specific goal.

## Architecture

The web app calls a FastAPI server (`api_server.py`) which wraps the Python pipeline. This ensures:
- **Single source of truth**: All intervention logic lives in Python
- **No code duplication**: TypeScript doesn't replicate Python logic
- **Easy updates**: Changes to the pipeline automatically apply to the web app
