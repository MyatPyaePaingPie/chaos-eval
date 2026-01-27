"""LLM extractors for document processing."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from .data_loader import Document
from .prompts import get_prompt_for_dataset, CHAOS_EXTRACTION_PROMPT

load_dotenv()


@dataclass
class ExtractionResult:
    """Result of an extraction attempt."""
    document_id: str
    model: str
    raw_response: str
    parsed_data: dict | None
    success: bool
    error: str | None = None
    latency_ms: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "model": self.model,
            "raw_response": self.raw_response,
            "parsed_data": self.parsed_data,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
        }


class BaseExtractor(ABC):
    """Base class for LLM extractors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def extract(self, document: Document, prompt: str | None = None) -> ExtractionResult:
        """Extract data from a document."""
        pass

    def extract_batch(self, documents: list[Document], prompt: str | None = None) -> list[ExtractionResult]:
        """Extract from multiple documents."""
        return [self.extract(doc, prompt) for doc in documents]

    def _parse_json_response(self, response: str) -> dict | None:
        """Try to parse JSON from response, handling common issues."""
        # Clean up response
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return None


class ClaudeExtractor(BaseExtractor):
    """Extract using Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        return self._client

    @property
    def model_name(self) -> str:
        return f"claude:{self.model}"

    def extract(self, document: Document, prompt: str | None = None) -> ExtractionResult:
        """Extract data using Claude's vision capabilities."""
        import time

        if prompt is None:
            prompt = get_prompt_for_dataset(document.dataset)

        try:
            start_time = time.time()

            # Prepare image
            image_data = document.to_base64()

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            latency_ms = (time.time() - start_time) * 1000
            raw_response = response.content[0].text
            parsed_data = self._parse_json_response(raw_response)

            return ExtractionResult(
                document_id=document.id,
                model=self.model_name,
                raw_response=raw_response,
                parsed_data=parsed_data,
                success=parsed_data is not None,
                latency_ms=latency_ms,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )

        except Exception as e:
            return ExtractionResult(
                document_id=document.id,
                model=self.model_name,
                raw_response="",
                parsed_data=None,
                success=False,
                error=str(e)
            )


class OllamaExtractor(BaseExtractor):
    """Extract using local models via Ollama."""

    def __init__(self, model: str = "llama3.2-vision"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )
        return self._client

    @property
    def model_name(self) -> str:
        return f"ollama:{self.model}"

    def extract(self, document: Document, prompt: str | None = None) -> ExtractionResult:
        """Extract data using Ollama's vision models."""
        import time

        if prompt is None:
            prompt = get_prompt_for_dataset(document.dataset)

        try:
            start_time = time.time()

            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [str(document.image_path)]
                    }
                ]
            )

            latency_ms = (time.time() - start_time) * 1000
            raw_response = response["message"]["content"]
            parsed_data = self._parse_json_response(raw_response)

            return ExtractionResult(
                document_id=document.id,
                model=self.model_name,
                raw_response=raw_response,
                parsed_data=parsed_data,
                success=parsed_data is not None,
                latency_ms=latency_ms,
                tokens_used=response.get("eval_count", 0)
            )

        except Exception as e:
            return ExtractionResult(
                document_id=document.id,
                model=self.model_name,
                raw_response="",
                parsed_data=None,
                success=False,
                error=str(e)
            )


class TesseractBaseline(BaseExtractor):
    """Tesseract OCR baseline for comparison."""

    @property
    def model_name(self) -> str:
        return "tesseract"

    def extract(self, document: Document, prompt: str | None = None) -> ExtractionResult:
        """Extract text using Tesseract OCR."""
        import time

        try:
            import pytesseract

            start_time = time.time()
            img = document.load_image()
            text = pytesseract.image_to_string(img)
            latency_ms = (time.time() - start_time) * 1000

            return ExtractionResult(
                document_id=document.id,
                model=self.model_name,
                raw_response=text,
                parsed_data={"extracted_text": text},
                success=bool(text.strip()),
                latency_ms=latency_ms
            )

        except Exception as e:
            return ExtractionResult(
                document_id=document.id,
                model=self.model_name,
                raw_response="",
                parsed_data=None,
                success=False,
                error=str(e)
            )


def get_extractor(model: str) -> BaseExtractor:
    """Factory function to get an extractor by model name."""
    model_lower = model.lower()

    if "claude" in model_lower:
        return ClaudeExtractor(model=model)
    elif "llama" in model_lower or "mistral" in model_lower or "qwen" in model_lower:
        return OllamaExtractor(model=model)
    elif "tesseract" in model_lower:
        return TesseractBaseline()
    else:
        # Default to Ollama for unknown models
        return OllamaExtractor(model=model)
