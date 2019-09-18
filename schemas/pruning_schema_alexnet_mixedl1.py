{
    ("features.0", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=1),
    ("features.3", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=1),
    ("features.6", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=1),
    ("features.8", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=1),
    ("features.10", "weight"): partial(prune.ln_structured, amount=0.2, axis=1, n=1),
    ("classifier.1", "weight"): partial(prune.l1_unstructured, amount=0.2),
    ("classifier.4", "weight"): partial(prune.l1_unstructured, amount=0.2),
    ("classifier.6", "weight"): partial(prune.l1_unstructured, amount=0.2),
}
