# from transformers import pipeline

# classifier = pipeline(
#     "zero-shot-classification",
#     model="facebook/bart-large-mnli",
#     device=0
# )

# LABELS = [
#     "multiple sclerosis medical question",
#     "general medical question",
#     "non medical question",
#     "unsafe medical question"
# ]


# def classify_query_hf(question: str):
#     result = classifier(question, LABELS)

#     top_label = result["labels"][0]
#     score = result["scores"][0]

#     if "multiple sclerosis" in top_label:
#         label = "ms_relevant"
#     elif "unsafe" in top_label:
#         label = "unsafe_medical"
#     elif "general medical" in top_label:
#         label = "medical_but_not_ms"
#     else:
#         label = "non_medical"

#     return {
#         "label": label,
#         "confidence": float(score),
#         "raw": result
#     }