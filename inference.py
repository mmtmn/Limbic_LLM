sample_texts = [
    "I'm feeling great today! The sun is shining, and everything seems to be falling into place.",
    "This is so frustrating. I can't believe how poorly everything is going.",
    "I'm utterly heartbroken. The day started so well, but it all fell apart."
]

sample_tokens = tokenizer(sample_texts, padding=True, truncation=True, max_length=32000, return_tensors="tf")
predictions = model.predict(sample_tokens['input_ids'])
predicted_emotions = label_encoder.inverse_transform(predictions.logits.argmax(axis=1).numpy())

for text, emotion in zip(sample_texts, predicted_emotions):
    print(f'Text: "{text}" => Predicted Emotion: {emotion}')
