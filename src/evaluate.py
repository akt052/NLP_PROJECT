from sacrebleu import corpus_bleu, corpus_chrf
import numpy as np

def compute_metrics(preds, refs):
    bleu = corpus_bleu(preds, [refs]).score / 100
    chrf = corpus_chrf(preds, [refs]).score / 100

    score = np.sqrt(bleu * chrf)
    return bleu, chrf, score