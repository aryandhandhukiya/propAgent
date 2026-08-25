"""
propOG AI Automation Intern task
---------------------------------
POST /analyze  { "description": "<raw listing text>" }
  -> headline, description, tags[3-5]
  -> bhk, property_type, locality, area_sqft (each: value or null)
  -> missing_fields: which of the 4 structured keys came back null

Run:
    export GEMINI_API_KEY=your_key_here   # optional — omit to use stub mode
    ./venv/bin/uvicorn app:app --reload --port 8000

Frontend: static/index.html (served at /)
"""

import asyncio
import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load local .env so GEMINI_API_KEY works without manual export.
load_dotenv()

app = FastAPI(title="propOG Listing Cleaner")

# Allow deployed frontend (Netlify) plus local dev frontends.
cors_origins = [
    "https://phenomenal-torte-5cf4d8.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

extra_cors_origins = os.environ.get("CORS_ORIGINS")
if extra_cors_origins:
    cors_origins.extend(
        [origin.strip() for origin in extra_cors_origins.split(",") if origin.strip()]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

REQUIRED_KEYS = ["bhk", "property_type", "locality", "area_sqft"]
ALL_KEYS = ["headline", "description", "tags"] + REQUIRED_KEYS
AMENITY_TAG_PATTERNS = [
    ("Gym", [r"\bgym\b", r"\bgymnasium\b"]),
    ("Parking", [r"\bparking\b", r"\bparking\s*lot\b", r"\bcar\s*park(?:ing)?\b"]),
    ("Swimming Pool", [r"\bswimming\s*pool\b", r"\bpool\b"]),
]


class ListingIn(BaseModel):
    description: str


PROMPT_TEMPLATE = """You are cleaning up a rushed, typo-ridden real-estate listing note written by a
field agent. You must respond with STRICT JSON ONLY — no markdown fences, no commentary before or
after the JSON.

Return exactly this shape:
{{
  "headline": "<short catchy headline>",
  "description": "<cleaned up 1-3 sentence description>",
  "tags": ["<3 to 5 short tags>"],
  "bhk": <integer number of bedrooms, or null>,
  "property_type": "<one of: flat, villa, plot, other, or null>",
  "locality": "<area/locality name as a string, or null>",
  "area_sqft": <integer square footage, or null>
}}

HARD RULE: Only use information that is EXPLICITLY stated in the raw text below. Do NOT guess,
infer, estimate, or invent a value for bhk, property_type, locality, or area_sqft. If the text does
not clearly state one of these, its value MUST be null. Do not round or "estimate" a nearby number
either — only use figures that are literally present in the text.

Raw listing text:
\"\"\"{raw_text}\"\"\"

Respond with the JSON object only.
"""


def build_prompt(raw_text: str) -> str:
    return PROMPT_TEMPLATE.format(raw_text=raw_text)


def strip_code_fences(text: str) -> str:
    """Gemini sometimes wraps JSON in ```json ... ``` even when told not to."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def call_gemini(raw_text: str) -> str:
    """Returns the raw text response from Gemini. Runs the blocking SDK call in a thread."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = build_prompt(raw_text)

    def _sync_call():
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return response.text

    return await asyncio.to_thread(_sync_call)


async def call_stub(raw_text: str) -> str:
    """Fallback used when no GEMINI_API_KEY is set. Waits ~1s like a real API call would,
    then returns a hardcoded response in the correct shape, doing a naive keyword-based
    extraction so the 'don't invent data' behavior is still demonstrably correct."""
    await asyncio.sleep(1)

    text_lower = raw_text.lower()

    bhk_match = re.search(r"(\d+)\s*bhk", text_lower)
    bhk = int(bhk_match.group(1)) if bhk_match else None

    property_type = None
    for candidate in ["flat", "villa", "plot"]:
        if candidate in text_lower:
            property_type = candidate
            break

    area_match = re.search(r"(\d{2,6})\s*sq\s*ft|(\d{2,6})\s*sqft", text_lower)
    area_sqft = None
    if area_match:
        area_sqft = int(area_match.group(1) or area_match.group(2))

    locality = None  # Deliberately not guessed — stub has no reliable way to detect this.

    result = {
        "headline": "Property listing (stub mode — no AI key configured)",
        "description": raw_text.strip().capitalize() + ".",
        "tags": ["property", "listing", "stub-mode"],
        "bhk": bhk,
        "property_type": property_type,
        "locality": locality,
        "area_sqft": area_sqft,
    }
    return json.dumps(result)


def validate_and_normalize(parsed: dict) -> dict:
    """Ensures all expected keys exist with roughly the right types. Raises ValueError on
    anything unrecoverable so the caller can retry."""
    if not isinstance(parsed, dict):
        raise ValueError("Response is not a JSON object")

    for key in ALL_KEYS:
        if key not in parsed:
            raise ValueError(f"Missing key: {key}")

    if not isinstance(parsed["headline"], str):
        raise ValueError("headline must be a string")
    if not isinstance(parsed["description"], str):
        raise ValueError("description must be a string")
    if not isinstance(parsed["tags"], list):
        raise ValueError("tags must be a list")

    if parsed["bhk"] is not None:
        try:
            parsed["bhk"] = int(parsed["bhk"])
        except (TypeError, ValueError):
            raise ValueError("bhk must be an integer or null")

    if parsed["property_type"] not in ("flat", "villa", "plot", "other", None):
        raise ValueError("property_type must be flat/villa/plot/other/null")

    if parsed["locality"] is not None and not isinstance(parsed["locality"], str):
        raise ValueError("locality must be a string or null")

    if parsed["area_sqft"] is not None:
        try:
            parsed["area_sqft"] = int(parsed["area_sqft"])
        except (TypeError, ValueError):
            raise ValueError("area_sqft must be an integer or null")

    # missing_fields is computed by us below, not trusted from the model — this guarantees
    # it's always mechanically correct regardless of what the model claims.
    parsed["missing_fields"] = [k for k in REQUIRED_KEYS if parsed.get(k) is None]

    return parsed


def enrich_tags_with_explicit_amenities(raw_text: str, tags: list) -> list:
    """Adds amenity tags when they are explicitly present in the raw listing text."""
    text = raw_text.lower()

    normalized_tags = []
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        clean = tag.strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized_tags.append(clean)

    for amenity_tag, patterns in AMENITY_TAG_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            key = amenity_tag.casefold()
            if key not in seen:
                seen.add(key)
                normalized_tags.append(amenity_tag)

    return normalized_tags


async def get_ai_response(raw_text: str) -> dict:
    caller = call_gemini if GEMINI_API_KEY else call_stub

    last_error = None
    for attempt in range(2):  # one initial try + one retry
        try:
            raw_response = await caller(raw_text)
            cleaned = strip_code_fences(raw_response)
            parsed = json.loads(cleaned)
            return validate_and_normalize(parsed)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            continue
        except Exception as e:
            # Keep backend failures as JSON HTTP errors instead of opaque 500s.
            last_error = e
            continue

    raise HTTPException(
        status_code=502,
        detail=f"AI response failed validation after retry: {last_error}",
    )


@app.post("/analyze")
async def analyze(listing: ListingIn):
    if not listing.description or not listing.description.strip():
        raise HTTPException(status_code=400, detail="description must not be empty")
    result = await get_ai_response(listing.description)
    result["tags"] = enrich_tags_with_explicit_amenities(
        listing.description,
        result.get("tags", []),
    )
    result["needs_more_info"] = len(result["missing_fields"]) > 2
    return result


@app.get("/api/health")
async def health():
    return {"status": "ok", "mode": "gemini" if GEMINI_API_KEY else "stub"}


# Serve the minimal frontend at /
app.mount("/", StaticFiles(directory="static", html=True), name="static")
