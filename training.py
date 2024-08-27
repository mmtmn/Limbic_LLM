from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model with gradient accumulation and data augmentation if needed
history = model.fit(
    X_train_tokens['input_ids'], 
    y_train, 
    validation_data=(X_test_tokens['input_ids'], y_test),
    epochs=10,
    batch_size=4,  # Small batch size due to large sequence length
    callbacks=[early_stopping, checkpoint]
)
