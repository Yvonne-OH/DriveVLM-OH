import tensorflow as tf
import pandas as pd
import tensorflow_text as text
from typing import List

class RougeLScorer:
    def __init__(self, alpha: float = 0.5):
        """
        Initialize the RougeLScorer with a specific alpha value.

        Args:
            alpha (float): Weight to balance precision and recall. Default is 0.5.
        """
        self.alpha = alpha

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """
        Tokenize the input text into words. Simple whitespace-based tokenizer.
        """
        return text.split()

    def compute_rouge_l(self, hypotheses: List[str], references: List[str]) -> dict:
        """
        Compute ROUGE-L score between hypotheses and references.

        Args:
            hypotheses (List[str]): List of generated sentences.
            references (List[str]): List of reference sentences.

        Returns:
            dict: Dictionary containing F-Measure, P-Measure, and R-Measure.
        """
        # Tokenize the input texts
        tokenized_hypotheses = [self.tokenize_text(hyp) for hyp in hypotheses]
        tokenized_references = [self.tokenize_text(ref) for ref in references]

        # Convert to ragged tensors
        ragged_hypotheses = tf.ragged.constant(tokenized_hypotheses)
        ragged_references = tf.ragged.constant(tokenized_references)

        # Compute ROUGE-L
        result = text.metrics.rouge_l(ragged_hypotheses, ragged_references, alpha=self.alpha)

        # Return the scores as a dictionary
        return {
            'F-Measure': result.f_measure.numpy(),
            'P-Measure': result.p_measure.numpy(),
            'R-Measure': result.r_measure.numpy()
        }

if __name__ == "__main__":
    # Example input
    generated_sentences = [
        "captain of the delta flight",
        "the 1990 transcript"
    ]

    reference_sentences = [
        "delta air lines flight",
        "this concludes the transcript"
    ]

    # Default ROUGE-L scorer
    scorer = RougeLScorer()

    # Default ROUGE-L scorer
    scorer = RougeLScorer()

    # Compute scores for different alpha values
    results = []

    # Compute scores for different alpha values
    results = []

    # Default alpha = 0.5
    scores = scorer.compute_rouge_l(generated_sentences, reference_sentences)
    results.append({"Alpha": 0.5, **scores})

    # Alpha = 0 (Recall focus)
    scorer.alpha = 0
    scores_alpha_0 = scorer.compute_rouge_l(generated_sentences, reference_sentences)
    results.append({"Alpha": 0, **scores_alpha_0})

    # Alpha = 1 (Precision focus)
    scorer.alpha = 1
    scores_alpha_1 = scorer.compute_rouge_l(generated_sentences, reference_sentences)
    results.append({"Alpha": 1, **scores_alpha_1})

    # Convert results to a DataFrame for tabular display
    df = pd.DataFrame(results)
    print("\nTabular Display of ROUGE-L Scores:")
    print(df.to_string(index=False))
