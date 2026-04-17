# ABAD AI Branding & Labeling Platform

AI-powered branding and product labeling tool for Azerbaijani small businesses. Enter a product name (e.g., "heyva mürəbbəsi") and the platform automatically generates market research, label design, regulatory compliance checks, and language quality analysis.

## Features

- **Product Analysis** — generates full product profile from a short input
- **Market Research** — competitor analysis, pricing, consumer trends for Azerbaijan
- **Design Recommendations** — color palette, typography, visual style suggestions
- **Label Text Generation** — professional Azerbaijani label copy with ingredients, nutrition, storage info
- **Visual Label Design** — AI-generated front label + structured back label with barcode
- **Regulatory Compliance** — checks against Azerbaijan Food Safety Law (Article 17.2.1), ГОСТ, AZS, ISO, AQTA
- **Language Quality** — spelling, grammar, and terminology review in Azerbaijani

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| AI Text Model | Google Gemini 3.1 Pro Preview |
| AI Image Model | Nano Banana Pro Preview |
| Image Processing | Pillow (PIL) |
| Frontend | Vanilla HTML/CSS/JS |
| Streaming | SSE (Server-Sent Events) |

## Setup

### Requirements

- Python 3.10, 3.11, or 3.12
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SananJafarov1/abad-ai-branding.git
cd abad-ai-branding
```

2. Create a `.env` file:
```bash
cp .env.example .env
```

3. Add your Gemini API key to `.env`:
```
GEMINI_API_KEY=your_gemini_api_key_here
```
Get your key from [Google AI Studio](https://aistudio.google.com/apikey)

4. Run the app:

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
bash run.sh
```

5. Open http://localhost:8001 in your browser

## How It Works

```
User Input ("heyva mürəbbəsi")
        │
        ▼
┌─────────────────────┐
│ Stage 1: Product    │──▶ Name, category, ingredients
│         Analysis    │
├─────────────────────┤
│ Stage 2: Market     │──▶ Competitors, pricing, trends
│         Research    │
├─────────────────────┤
│ Stage 3: Design     │──▶ Colors, typography, visual style
│    Recommendations  │
├─────────────────────┤
│   User confirms     │──▶ Edit color palette
│   color palette     │
├─────────────────────┤
│ Stage 4: Label Text │──▶ Product name, tagline, nutrition
│       Generation    │
├─────────────────────┤
│ Stage 5: Visual     │──▶ Front label (AI) + Back label (Pillow)
│         Label       │
├─────────────────────┤
│ Stage 6: Compliance │──▶ Law, ГОСТ, AZS, ISO, AQTA checks
│         Check       │
├─────────────────────┤
│ Stage 7: Language   │──▶ Spelling, grammar, terminology
│         Check       │
└─────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend |
| `/health` | GET | Health check |
| `/api/analyze` | POST | Phase 1: product analysis, market research, design |
| `/api/analyze/continue` | POST | Phase 2: label text, visual, compliance, language |

## License

This project was developed for the ASAN AI Hub Challenge by ABAD.
