import json
from typing import List, Dict, Union, Optional
import numpy as np

from others.logging import init_logger

logger = init_logger()

def summarize_multi_rewrite_quality(
    results: Union[List[List[Dict]], str]
):
    efficacy, paraphrase, specificity = [], [], []
    for result in results:
        metrics = summarize_rewrite_quality(result, False)
        efficacy += [metrics[0]]
        paraphrase += [metrics[1]]
        specificity += [metrics[2]]

    logger.info(
        "Evaluation Results: "
        "Score: {:.2f} "
        "Efficacy: {:.2f} "
        "Paraphrase: {:.2f} "
        "Specificity: {:.2f} "
        .format(
            100/3 * (efficacy + paraphrase + specificity),
            100 * np.mean(efficacy),
            100 * np.mean(paraphrase),
            100 * np.mean(specificity),
        )
    )


def summarize_rewrite_quality(
    results: Union[List[Dict], str],
    report: Optional[bool] = True,
):
    if isinstance(results, str):
        with open(results, "r") as f:
            results = json.load(f)

    efficacy, paraphrase, specificity = [], [], []

    for line in results:
        efficacy += [np.mean(line["rewrite_prompts_correct"])]
        paraphrase += [np.mean(line["paraphrase_prompts_correct"])]
        specificity += [np.mean(line["neighborhood_prompts_correct"])]
    
    efficacy = np.mean(efficacy)
    paraphrase = np.mean(paraphrase)
    specificity = np.mean(specificity)
    
    if report:
        logger.info(
            "Evaluation Results: "
            "Score: {:.2f} "
            "Efficacy: {:.2f} "
            "Paraphrase: {:.2f} "
            "Specificity: {:.2f} "
            .format(
                100/3 * (efficacy + paraphrase + specificity),
                100 * efficacy,
                100 * paraphrase,
                100 * specificity,
            )
        )

    return efficacy, paraphrase, specificity