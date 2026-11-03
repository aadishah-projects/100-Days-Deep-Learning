# Customer Churn — Project Learning Note

Folder: [Customer Churn](<F:\100 Days Deep Learnign\Customer Churn>)  
Main notebook: [ann.ipynb](<F:\100 Days Deep Learnign\Customer Churn\ann.ipynb>)

## 1. What this project does

This is a supervised binary-classification project.

The model receives customer information and predicts:

- `0` → customer stays
- `1` → customer exits

The model’s final output is a probability such as `0.82`, meaning approximately an 82% predicted chance of churn.

The dataset contains 10,000 customers. About 20.37% actually exited, so the classes are imbalanced.

## 2. Complete pipeline

Your notebook follows the standard machine-learning pipeline:

```text
Raw data
   ↓
Remove irrelevant columns
   ↓
Encode categorical values
   ↓
Separate X and y
   ↓
Train/test split
   ↓
Feature scaling
   ↓
Build neural network
   ↓
Train with backpropagation
   ↓
Predict probabilities
   ↓
Convert probabilities into classes
   ↓
Evaluate performance
```

This pipeline is extremely important because most real AI projects are built around this same structure.

## 3. Data preprocessing

### Removing columns

You removed:

```python
RowNumber
CustomerId
Surname
```

These columns identify the customer but do not meaningfully describe churn behavior. Keeping them could make the model memorize meaningless patterns.

### One-hot encoding

You converted categorical columns such as `Geography` and `Gender` into numerical columns:

```text
Geography_Germany
Geography_Spain
Gender_Male
```

Because `drop_first=True`, France and Female become the reference categories.

For example:

```text
Geography_Germany = 0
Geography_Spain = 0
```

means the customer is from France.

### Feature scaling

You used `StandardScaler`, which transforms each numerical feature approximately as:

\[
z = \frac{x-\mu}{\sigma}
\]

This gives features similar scales, helping gradient descent learn more effectively.

Important correction:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

The second line should be:

```python
X_test_scaled = scaler.transform(X_test)
```

You must fit the scaler only on training data. Fitting again on test data allows information from the test set to influence preprocessing and makes evaluation less trustworthy.

## 4. Understanding the neural network

Your network is:

```text
11 input features
      ↓
Dense layer: 11 neurons + ReLU
      ↓
Dense layer: 11 neurons + ReLU
      ↓
Dense layer: 1 neuron + Sigmoid
      ↓
Churn probability
```

The model has 276 trainable parameters.

A dense layer performs:

\[
z = xW+b
\]

Then the activation function transforms the result.

### ReLU

\[
\text{ReLU}(x)=\max(0,x)
\]

ReLU helps the network learn nonlinear relationships.

### Sigmoid

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Sigmoid converts the final value into a number between 0 and 1, which is suitable for binary classification.

## 5. Loss, optimizer, and training

You used:

```python
loss="binary_crossentropy"
optimizer="Adam"
```

Binary cross-entropy measures how far the predicted probability is from the actual answer.

For one example:

\[
L=-[y\log(\hat y)+(1-y)\log(1-\hat y)]
\]

Adam updates the weights using gradients calculated through backpropagation.

During each epoch:

1. The model makes predictions.
2. The loss is calculated.
3. Backpropagation calculates gradients.
4. Adam updates the weights.
5. The process repeats.

Your test accuracy was approximately:

```text
86.1%
```

However, accuracy alone is not enough. Since 79.63% of customers did not churn, a model that always predicts “stays” would already achieve about 79.63% accuracy.

You should also measure:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- PR-AUC

For churn prediction, recall may be especially important because missing a customer who is likely to leave can be costly.

## 6. Important learning insight

Your model does not directly predict a class. It predicts a probability:

```python
y_log = model.predict(X_test_scaled)
```

Then you apply a threshold:

```python
y_pred = np.where(y_log > 0.5, 1, 0)
```

Changing the threshold changes the business behavior:

- Lower threshold → catch more possible churners, but more false alarms.
- Higher threshold → fewer false alarms, but possibly miss real churners.

In real AI engineering, the threshold should be selected according to the cost of different mistakes.

## 7. Issues and improvements

For a stronger implementation:

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential([
    Input(shape=(11,)),
    Dense(11, activation="relu"),
    Dense(11, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

Also consider:

- Use `stratify=y` in `train_test_split`.
- Add early stopping instead of blindly training for 100 epochs.
- Plot loss and accuracy in separate graphs.
- Compare against logistic regression and XGBoost.
- Use class weights if minority-class recall is poor.
- Save the trained model and scaler together.
- Build an inference function that applies exactly the same preprocessing to new customers.

## 8. Connection to your LLM goal

This project teaches foundations used inside larger models:

- Dense layers are the basic linear transformations used throughout Transformers.
- Activations introduce nonlinearity.
- Backpropagation trains nearly all neural networks.
- Optimizers such as Adam are also used in LLM training.
- Data preprocessing and evaluation are just as important as model architecture.
- Saving preprocessing along with the model is essential for production inference.

An LLM is much larger and uses attention, but the fundamental learning loop remains:

```text
input → forward pass → loss → backward pass → optimizer update
```

## 9. Experiments worth doing

To truly understand this project, try:

1. Fix the scaler bug and compare the result.
2. Test thresholds `0.3`, `0.5`, and `0.7`.
3. Generate a confusion matrix and classification report.
4. Add early stopping.
5. Change the hidden-layer sizes.
6. Compare the ANN with logistic regression.
7. Remove one feature at a time and observe the effect.
8. Create a function that predicts churn for one new customer.

The main lesson is not that this particular ANN is the best churn model. The important lesson is that you built a complete neural-network workflow from raw tabular data to model evaluation.