import random

# Expanded mock dataset (100 examples)
emotions = ['joy', 'anger', 'sadness']
texts = [
    "I am so happy today!",
    "This is a terrible situation.",
    "I feel so sad and lonely.",
    "What a wonderful surprise!",
    "I am angry at you.",
    "This is the best day ever!",
    "Life is beautiful and full of joy.",
    "Why does everything always go wrong?",
    "I'm filled with sadness.",
    "Happiness is all around me.",
    "I am frustrated beyond belief.",
    "This is the most amazing thing I've ever seen!",
    "I can't take this anymore.",
    "My heart is broken.",
    "I'm excited about the future!",
    "Everything is falling apart.",
    "This brings me so much joy!",
    "I can't stand this situation any longer.",
    "I am devastated by the news.",
    "This day couldn't get any better!",
    "I'm so disappointed in how things turned out.",
    "I am so proud of what I've accomplished.",
    "Why is life so unfair?",
    "I feel hopeless and lost.",
    "This moment is perfect!",
    "I am so mad right now!",
    "I can't stop crying.",
    "I am overjoyed with the results!",
    "This is such a mess.",
    "I'm drowning in sorrow.",
    "This is the best feeling in the world!",
    "I'm so fed up with everything.",
    "I can't believe this is happening.",
    "I'm on top of the world!",
    "This is completely unacceptable.",
    "I feel empty inside.",
    "I can't wait to see what happens next!",
    "Everything is ruined.",
    "I'm overwhelmed with sadness.",
    "This is a dream come true!",
    "I'm so irritated right now.",
    "I feel like I'm falling apart.",
    "I am ecstatic about the news!",
    "Nothing ever goes my way.",
    "I can't shake this feeling of sadness.",
    "I am full of joy!",
    "I'm so angry I could scream.",
    "I feel so down today.",
    "This is the happiest day of my life!",
    "I am furious with the outcome.",
    "I am overcome with grief.",
    "I am bursting with happiness!",
    "I can't believe how badly this went.",
    "I'm heartbroken beyond words.",
    "I am thrilled with how things turned out!",
    "Why is everything so difficult?",
    "I feel like there's no hope left.",
    "I am so grateful for this moment!",
    "I'm livid about the situation.",
    "I can't stop thinking about how sad I am.",
    "This is the most joy I've ever felt!",
    "I'm so annoyed with everything.",
    "I feel like I'm in a dark place.",
    "I am so excited for what's to come!",
    "I'm angry and I don't know what to do.",
    "I feel like the world is against me.",
    "I am so happy I could cry!",
    "This situation makes me so angry.",
    "I can't escape this sadness.",
    "I am so proud of myself!",
    "I'm furious and disappointed.",
    "I can't seem to find any joy.",
    "I am so excited about this opportunity!",
    "Everything is falling apart around me.",
    "I feel so hopeless and alone.",
    "I am over the moon with excitement!",
    "I'm so frustrated with everything.",
    "I feel like nothing will ever be okay again.",
    "I am so full of joy today!",
    "This is the worst day ever.",
    "I feel like I can't go on.",
    "I am so excited about the possibilities!",
    "I am enraged by this outcome.",
    "I feel like my world is ending.",
    "I am so happy I can't stop smiling!",
    "This is the last straw.",
    "I feel like I'm sinking into despair.",
    "I am so proud of what we've achieved!",
    "I'm angry beyond words.",
    "I feel like I'm drowning in sorrow.",
    "I am so thrilled with the results!",
    "Why does nothing ever work out?",
    "I feel so sad and lost.",
    "I am so happy with how everything turned out!",
    "I'm angry and frustrated with everything.",
    "I can't seem to find any happiness.",
    "I am so grateful for this success!"
]

# Generate 100 random pairs of text and emotion
data = {
    'text': random.choices(texts, k=100),
    'emotion': random.choices(emotions, k=100)
}

df = pd.DataFrame(data)

# Encode the emotions into integers
df['emotion'] = label_encoder.fit_transform(df['emotion'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['emotion'], test_size=0.2, random_state=42)

# Tokenize the text with a max length of 32k
X_train_tokens = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=32000, return_tensors="tf")
X_test_tokens = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=32000, return_tensors="tf")

# Generate 100 sample texts for inference
sample_texts = random.choices(texts, k=100)

# Tokenize these sample texts
sample_tokens = tokenizer(sample_texts, padding=True, truncation=True, max_length=32000, return_tensors="tf")


# Predict emotions for the 100 sample texts
predictions = model.predict(sample_tokens['input_ids'])
predicted_emotions = label_encoder.inverse_transform(predictions.logits.argmax(axis=1).numpy())

# Print the predicted emotions for the sample texts
for text, emotion in zip(sample_texts, predicted_emotions):
    print(f'Text: "{text}" => Predicted Emotion: {emotion}")
