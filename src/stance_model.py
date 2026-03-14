"""
Wrapper for the Political DEBATE zero-shot NLI model for stance detection.

The Political DEBATE model (Burnham et al. 2025) uses a DeBERTa architecture
trained for entail/not-entail classification on political text. We use it
to classify articles as pro-gold, pro-silver, both, or neither.

The model works by pairing a premise (the article text) with a hypothesis
(e.g., "This article supports the gold standard") and returning entailment
probabilities.
"""

from typing import Optional
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Default hypothesis statements for the two stance dimensions
# ---------------------------------------------------------------------------
# These should be kept short and simple per the Political DEBATE best practices.
# We run two separate binary classifiers as discussed.

DEFAULT_HYPOTHESES = {
    "pro_gold": "This text supports the gold standard.",
    "pro_silver": "This text supports the free coinage of silver.",
}


class StanceDetector:
    """
    Wrapper around the Political DEBATE model for stance detection.

    Uses the zero-shot-classification pipeline for convenience, or the
    raw model for more control over batching and performance.
    """

    def __init__(
        self,
        model_name: str = "mlburnham/Political_DEBATE_large_v1.0",
        device: Optional[str] = None,
        use_pipeline: bool = True,
        max_length: int = 512,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model identifier.
        device : str, optional
            "cuda", "cpu", or None (auto-detect).
        use_pipeline : bool
            If True, use the HF zero-shot-classification pipeline.
            If False, use the raw model for more control.
        max_length : int
            Max token length for input truncation.
        """
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")

        if use_pipeline:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1,
            )
            self.tokenizer = None
            self.model = None
        else:
            self.classifier = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)
            self.model.eval()

        print("Model loaded.")

    def classify_single(
        self,
        text: str,
        hypothesis: str,
    ) -> dict:
        """
        Classify a single text against a single hypothesis.

        Returns
        -------
        dict with keys:
            - 'entailment_score': float, probability of entailment
            - 'not_entailment_score': float, probability of not-entailment
        """
        if self.classifier is not None:
            # Pipeline approach
            result = self.classifier(
                text,
                candidate_labels=[hypothesis],
                hypothesis_template="{}",
            )
            score = result["scores"][0]
            return {
                "entailment_score": score,
                "not_entailment_score": 1 - score,
            }
        else:
            # Raw model approach
            inputs = self.tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Political DEBATE uses entail/not-entail (2 classes)
            # Typically index 0 = entailment, but verify with model config
            entail_idx = 0
            not_entail_idx = 1

            return {
                "entailment_score": probs[0][entail_idx].item(),
                "not_entailment_score": probs[0][not_entail_idx].item(),
            }

    def detect_stances(
        self,
        text: str,
        hypotheses: Optional[dict] = None,
    ) -> dict:
        """
        Run both pro-gold and pro-silver stance detection on a single article.

        Parameters
        ----------
        text : str
            The article text (premise).
        hypotheses : dict, optional
            Dict with keys 'pro_gold' and 'pro_silver' mapping to hypothesis
            strings. Defaults to DEFAULT_HYPOTHESES.

        Returns
        -------
        dict with keys:
            - 'pro_gold_score': float
            - 'pro_silver_score': float
        """
        if hypotheses is None:
            hypotheses = DEFAULT_HYPOTHESES

        gold_result = self.classify_single(text, hypotheses["pro_gold"])
        silver_result = self.classify_single(text, hypotheses["pro_silver"])

        return {
            "pro_gold_score": gold_result["entailment_score"],
            "pro_silver_score": silver_result["entailment_score"],
        }

    def detect_stances_batch(
        self,
        df: pd.DataFrame,
        text_column: str = "article",
        hypotheses: Optional[dict] = None,
        max_text_length: int = 2000,
    ) -> pd.DataFrame:
        """
        Apply stance detection to all articles in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with article text.
        text_column : str
            Name of the column containing article text.
        hypotheses : dict, optional
            Hypothesis strings for pro-gold and pro-silver.
        max_text_length : int
            Truncate article text to this many characters before tokenization.
            This is a pre-tokenizer safety limit; the tokenizer also truncates.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added columns: 'pro_gold_score',
            'pro_silver_score'.
        """
        if hypotheses is None:
            hypotheses = DEFAULT_HYPOTHESES

        gold_scores = []
        silver_scores = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Stance detection"):
            text = str(row[text_column])[:max_text_length]

            try:
                scores = self.detect_stances(text, hypotheses)
                gold_scores.append(scores["pro_gold_score"])
                silver_scores.append(scores["pro_silver_score"])
            except Exception as e:
                print(f"Error on article {row.get('article_id', '?')}: {e}")
                gold_scores.append(None)
                silver_scores.append(None)

        df = df.copy()
        df["pro_gold_score"] = gold_scores
        df["pro_silver_score"] = silver_scores

        return df
