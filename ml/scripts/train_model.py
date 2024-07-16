# Import necessary libraries
def train_model(X_train, y_train, model):
    """Train the model."""
    model.fit(X_train, y_train)
    return model
