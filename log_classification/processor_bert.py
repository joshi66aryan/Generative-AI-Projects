import joblib

_transformer_model = None
_classifier_model = None


def _load_models():
    global _transformer_model, _classifier_model
    if _transformer_model is None:
        from sentence_transformers import SentenceTransformer

        _transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    if _classifier_model is None:
        _classifier_model = joblib.load("models/log_classifier.joblib")
    return _transformer_model, _classifier_model

def classify_with_bert(log_message):
    try:
        transformer_model, classifier_model = _load_models()
        message_embeddings = transformer_model.encode(log_message)
        probabilities = classifier_model.predict_proba([message_embeddings])[0]
    except Exception:
        return "Unclassified"
    if max(probabilities) < 0.5:
        return "Unclassified"
    predicted_class = classifier_model.classes_[probabilities.argmax()]
    return predicted_class

if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)
