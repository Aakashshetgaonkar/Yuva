import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

(x_train_full, y_train_full), (x_test, y_test) = (tf.keras.datasets.fashion_mnist.load_data())

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", 
    "Coat", "Sandal", "Shirt", "Sneaker","Bag", "Ankle boot"]

print("Training data:", x_train_full.shape)
print("Testing data :", x_test.shape)

# Normalize pixel values from 0-255 to 0-1
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full,y_train_full,test_size=0.15,
    random_state=SEED,stratify=y_train_full)

print("\nTraining samples  :", len(x_train))
print("Validation samples:", len(x_val))
print("Testing samples   :", len(x_test))


plt.figure(figsize=(12, 6))

for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.suptitle("Sample Fashion-MNIST Images")
plt.tight_layout()
plt.show()

# 5. CALLBACKS

def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss",patience=4,restore_best_weights=True,verbose=1),
        ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2,min_lr=1e-6,verbose=1)
    ]

# 6. MODEL 1: BASIC ANN

def build_model_1():

    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ], name="Basic_ANN")

    model.compile(optimizer="sgd",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    return model

# 7. MODEL 2: DEEPER ANN + ADAM

def build_model_2():

    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ], name="Deeper_ANN")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    return model

# 8. MODEL 3: ANN + BATCH NORMALIZATION + DROPOUT

def build_model_3():

    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(256, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.30),
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ], name="ANN_BN_Dropout")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    return model


# 9. TRAINING FUNCTION

EPOCHS = 20
BATCH_SIZE = 128

histories = {}
trained_models = {}
results = {}

def train_and_evaluate(model, name):

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.summary()

    history = model.fit(x_train,y_train,validation_data=(x_val, y_val),
        epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=get_callbacks(),verbose=1)

    # Evaluate on held-out test data
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Save model and training history
    histories[name] = history
    trained_models[name] = model

    results[name] = {"Test Accuracy": test_accuracy,"Test Loss": test_loss,
        "Epochs Trained": len(history.history["loss"])}

    print(f"\n{name} Test Accuracy: {test_accuracy:.4f}")
    print(f"{name} Test Loss: {test_loss:.4f}")

    return model


# 10. TRAIN MODELS

model1 = build_model_1()
model1 = train_and_evaluate(model1,"Model 1 - Basic ANN")

model2 = build_model_2()
model2 = train_and_evaluate(model2,"Model 2 - Deeper ANN + Adam")

model3 = build_model_3()
model3 = train_and_evaluate(model3,"Model 3 - ANN + BN + Dropout")

results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
print("\nFINAL MODEL COMPARISON")
print(results_df)

# Accuracy comparison graph
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index,y=results_df["Test Accuracy"])
plt.title("Fashion-MNIST ANN Model Accuracy Comparison")
plt.ylabel("Test Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 14. VALIDATION ACCURACY COMPARISON

plt.figure(figsize=(10, 6))

for name, history in histories.items():
    plt.plot(history.history["val_accuracy"],label=name)

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 15. VALIDATION LOSS COMPARISON

plt.figure(figsize=(10, 6))
for name, history in histories.items():
    plt.plot(history.history["val_loss"],label=name)

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 16. SELECT MODEL USING VALIDATION ACCURACY

best_model_name = max(histories,key=lambda name: max(histories[name].history["val_accuracy"]))
best_model = trained_models[best_model_name]
print("\nSelected model:", best_model_name)

# Final evaluation on test set
test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)

print("Final Test Accuracy:", round(test_accuracy, 4))
print("Final Test Loss:", round(test_loss, 4))

# 17. CLASSIFICATION REPORT

y_prob = best_model.predict(x_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_test,y_pred,target_names=class_names,digits=4))

# 18. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(11, 8))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=class_names,yticklabels=class_names)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 19. DISPLAY MISCLASSIFIED IMAGES

incorrect_indices = np.where(y_pred != y_test)[0]
plt.figure(figsize=(12, 6))
for i, idx in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(f"True: {class_names[y_test[idx]]}\n"f"Pred: {class_names[y_pred[idx]]}",fontsize=8)
    plt.axis("off")

plt.suptitle("Misclassified Test Images")
plt.tight_layout()
plt.show()

# 20. DISPLAY SAMPLE PREDICTIONS

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Predicted: {class_names[y_pred[i]]}\n"f"Actual: {class_names[y_test[i]]}",fontsize=8)
    plt.axis("off")

plt.suptitle("Sample Predictions")
plt.tight_layout()
plt.show()

#21. SAVE RESULTS AND BEST MODEL

results_df.to_csv("fashion_mnist_ann_results.csv")
best_model.save("best_fashion_mnist_ann.keras")
print("\nResults saved to fashion_mnist_ann_results.csv")
print("Best ANN model saved successfully.")