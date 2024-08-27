from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf

# Load a pre-trained transformer model for sequence classification
model = TFAutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=len(df['emotion'].unique()))

# Compile the model with Adam optimizer and learning rate scheduling
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Display model architecture
model.summary()
