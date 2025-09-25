import torch

def predict_fake_news(model, tokenizer, device, texts, max_length=256):
    model.eval()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        predictions = torch.argmax(logits, dim=-1)

    label_map = {0: "FAKE", 1: "REAL"}
    return [label_map[p.item()] for p in predictions]
