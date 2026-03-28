from typing import List, Dict
import os

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    PYCOCOEVALCAP_AVAILABLE = True
except ImportError:
    PYCOCOEVALCAP_AVAILABLE = False
    print("Warning: pycocoevalcap not available. Install with: pip install git+https://github.com/salaniz/pycocoevalcap.git")


class CaptionEvaluator:
    
    def __init__(self):
        if not PYCOCOEVALCAP_AVAILABLE:
            raise ImportError(
                "pycocoevalcap is not available. "
                "Install with: pip install git+https://github.com/salaniz/pycocoevalcap.git"
            )
        
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
    
    def compute_metrics(
        self, 
        ref_list: List[List[str]], 
        hyp_list: List[str]
    ) -> Dict[str, float]:
        # Convert to COCO format: {id: [captions]}
        gts = {}  # ground truths
        res = {}  # predictions
        
        # Handle different ref_list formats
        if isinstance(ref_list[0], list):
            # Multiple references per sample (e.g., MSRVTT with 20 refs per video)
            for i, hyp in enumerate(hyp_list):
                gts[i] = [ref_list[j][i] for j in range(len(ref_list))]
                res[i] = [hyp]
        else:
            # Single reference per sample (e.g., YoucookII)
            for i, (ref, hyp) in enumerate(zip(ref_list, hyp_list)):
                gts[i] = [ref] if isinstance(ref, str) else ref
                res[i] = [hyp]
        
        # Compute scores
        output = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                # BLEU returns multiple scores
                for m, s in zip(method, score):
                    output[m] = s
            else:
                output[method] = score
        
        return output
