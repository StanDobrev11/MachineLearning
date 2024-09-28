import numpy as np
from scipy.optimize import minimize


class CustomBaseRegression:
    def __init__(self):
        self._coef = None  # Internal attribute to store the coefficients
        self._intercept = None

    @property
    def intercept_(self):
        return self._intercept

    @intercept_.setter
    def intercept_(self, value):
        if value is not None:
            self._intercept = value  # Update the internal attribute

    @property
    def coef_(self):
        """Getter for coef_"""
        return self._coef

    @coef_.setter
    def coef_(self, value):
        """Setter for coef_"""
        if value is not None:
            self._coef = value  # Update the internal attribute

    @staticmethod
    def _linear_combination(features, slopes, interceptor):
        """Compute the linear combination of features and slopes with intercept."""
        # Use np.dot to compute the dot product between slopes and features
        result_sum = np.dot(features, slopes)

        # Add the intercept term to each prediction
        return result_sum + interceptor

    @staticmethod
    def _generate_slopes(number, random_range=(-1, 1)):
        # generates initial slopes and are in range (-1; 1)
        return [np.random.randint(*random_range) for _ in range(number)]


class CustomLinearRegression(CustomBaseRegression):

    def predict(self, features):
        """Make predictions using the learned coefficients and intercept."""
        return np.dot(features, np.array(self.coef_), ) + self.intercept_

    def score(self, features, y_real):
        """Calculate the R-squared score."""
        y_pred = self.predict(features)
        return 1 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - y_real.mean()) ** 2)

    def fit(self, features, target, alpha=0.01, iterations=1000, random_state=None):
        """Perform fit with gradient descent."""

        # Set the random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # initialize randomly
        slopes = np.random.uniform(-1, 1, size=features.shape[1])  # Default random initialization

        interceptor = 0  # Fixed interceptor

        # Perform gradient descent for the specified number of iterations
        for _ in range(iterations):
            slopes, interceptor = self._gradient_descent(features, target, slopes, interceptor, alpha)

        # Store the learned coefficients and intercept
        self.coef_ = np.array(slopes)
        self.intercept_ = interceptor

        # Return self for method chaining (similar to scikit-learns API)
        return self

    def _parameter_gradient(self, features, target, slopes, interceptor):
        """Compute gradients for the parameters using Mean Squared Error (MSE)."""

        # Calculate the predicted values (linear combination) just once
        predictions = self._linear_combination(features, slopes, interceptor)

        # Calculate the residuals (errors between the target and predictions)
        residuals = target - predictions

        # Calculate the gradients for the slopes (one per feature) using vectorized operations
        gradients = -2 / len(target) * np.dot(features.T, residuals)

        # Calculate the gradient for the intercept (bias term)
        interceptor_gradient = -2 / len(target) * np.sum(residuals)

        return gradients, interceptor_gradient

    def _gradient_descent(self, features, target, slopes, interceptor, alpha):
        """Perform one step of gradient descent to update slopes and intercept."""

        # Get the gradients for slopes and intercept
        gradients, interceptor_gradient = self._parameter_gradient(features, target, slopes, interceptor)

        # Update slopes using vectorized operations
        slopes_new = np.array(slopes) - alpha * gradients

        # Update intercept
        interceptor_new = interceptor - alpha * interceptor_gradient

        return slopes_new, interceptor_new


class CustomLogisticRegression(CustomBaseRegression):

    def _parameter_gradient(self, features, target, slopes, interceptor):
        """Compute the gradient of cross-entropy loss with respect to slopes and intercept."""
        # Calculate the predicted probabilities using sigmoid
        linear_combination = self._linear_combination(features, slopes, interceptor)
        y_pred = self._sigmoid(linear_combination)

        # Calculate the residuals (errors between predicted probabilities and actual target)
        residuals = y_pred - target

        # Gradient for slopes: using features and residuals
        gradients = np.dot(features.T, residuals) / len(target)

        # Gradient for intercept (bias term)
        interceptor_gradient = np.mean(residuals)

        return gradients, interceptor_gradient

    def fit(self, features, target, alpha=0.01, iterations=1000, random_state=None):
        """Perform fit using gradient descent and cross-entropy loss."""
        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # Initialize parameters (slopes and intercept) randomly
        initial_params = np.random.uniform(-0.01, 0.01, size=features.shape[1] + 1)

        # Minimize the cross-entropy loss using L-BFGS-B solver
        result = minimize(
            fun=self._cross_entropy_loss,
            x0=initial_params,
            args=(features, target),
            method='L-BFGS-B'
        )

        # The result contains optimized parameters (slopes and intercept)
        self.coef_ = result.x[:-1]  # All but last element are slopes
        self.intercept_ = result.x[-1]  # Last element is intercept

        return self

    def _cross_entropy_loss(self, params, features, target):
        """Compute cross-entropy loss and its gradients."""
        # Split params into slopes and intercept
        slopes = params[:-1]  # All but the last param
        intercept = params[-1]  # Last param is the intercept

        # Get predicted probabilities using sigmoid
        y_pred = self._sigmoid(np.dot(features, slopes) + intercept)

        # Compute cross-entropy loss
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(target * np.log(y_pred) + (1 - target) * np.log(1 - y_pred))

        return loss

    @staticmethod
    def _sigmoid(z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, features):
        """Make probability predictions using the learned coefficients and intercept."""
        # Calculate the linear combination (z)
        z = np.dot(features, np.array(self.coef_)) + self.intercept_

        # Apply the sigmoid function to get probabilities
        return self._sigmoid(z)

    def predict_class(self, features):
        """Predict class (0 or 1) based on a threshold of 0.5."""
        prob = self.predict_proba(features)
        return np.where(prob >= 0.5, 1, 0)

    def score(self, features, target):
        """Compute the accuracy of the model on the given dataset."""
        # Predict the class labels using the trained model
        y_pred = self.predict_class(features)

        # Calculate accuracy by comparing the predicted labels with the true labels
        accuracy = np.mean(y_pred == target)

        return accuracy
