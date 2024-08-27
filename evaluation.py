# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test_tokens['input_ids'], y_test)
print(f'Test Accuracy: {accuracy:.2f}')
