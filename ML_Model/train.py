import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import signal

WINDOW_SIZE = 100
NUM_FEATURES = 6
BATCH_SIZE = 16
EPOCHS = 100
RANDOM_SEED = 42

# ============================================
# DATA AUGMENTATION (identique)
# ============================================

def augment_window(window):
    """Applique des transformations aléatoires"""
    augmented = []
    augmented.append(window.copy())

    # Bruit
    noise = np.random.normal(0, 0.05, window.shape)
    augmented.append(window + noise)

    # Scaling
    scale = np.random.uniform(0.8, 1.2)
    augmented.append(window * scale)

    # Time warping
    new_length = np.random.randint(90, 110)
    warped = signal.resample(window, new_length, axis=0)
    if len(warped) < 100:
        padded = np.zeros((100, window.shape[1]))
        padded[:len(warped)] = warped
        augmented.append(padded)
    else:
        augmented.append(warped[:100])

    # Time shift
    shift = np.random.randint(-10, 10)
    shifted = np.roll(window, shift, axis=0)
    augmented.append(shifted)

    # Rotation
    rotated = window.copy()
    angle = np.random.uniform(-0.2, 0.2)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotated[:, 3] = window[:, 3] * cos_a - window[:, 4] * sin_a
    rotated[:, 4] = window[:, 3] * sin_a + window[:, 4] * cos_a
    augmented.append(rotated)

    return augmented

def augment_dataset(X, y, augmentation_factor=6):
    """Augmente le dataset"""
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        augmented_windows = augment_window(X[i])
        num_to_take = min(augmentation_factor, len(augmented_windows))

        for j in range(num_to_take):
            X_augmented.append(augmented_windows[j])
            y_augmented.append(y[i])

    return np.array(X_augmented, dtype=np.float32), np.array(y_augmented)

# ============================================
# HELPER FUNCTIONS
# ============================================

def load_and_preprocess_data(file_path, window_size=WINDOW_SIZE):
    print(f"Chargement depuis {file_path}...")
    data = pd.read_csv(file_path, delimiter=';')

    class_mapping = {}
    for class_id in data['class_id'].unique():
        class_name = data[data['class_id'] == class_id]['class_name'].iloc[0]
        class_mapping[class_id] = class_name

    grouped = data.groupby(['window_id', 'class_id'])
    x_recordings = []
    y_recordings = []

    for (window_id, class_id), group in grouped:
        group = group.sort_values('timestamp')
        features = group[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values

        if len(features) >= window_size:
            x_recordings.append(features[:window_size, :])
            y_recordings.append(class_id)
        elif len(features) >= int(window_size * 0.8):
            padded = np.zeros((window_size, NUM_FEATURES))
            padded[:len(features), :] = features
            x_recordings.append(padded)
            y_recordings.append(class_id)

    return np.array(x_recordings, dtype=np.float32), np.array(y_recordings), class_mapping

def normalize_data(x_data):
    normalized = np.zeros_like(x_data, dtype=np.float32)
    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[2]):
            channel_data = x_data[i, :, j]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:
                normalized[i, :, j] = (channel_data - mean) / std
            else:
                normalized[i, :, j] = channel_data - mean
    return normalized

def create_minimal_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(8, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),  # Augmenté de 0.4 à 0.5
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ============================================
# MAIN - VALIDATION CORRECTE
# ============================================

print("="*60)
print("ENTRAÎNEMENT AVEC VALIDATION CORRECTE")
print("="*60)

# 1. Charger données originales
file_path = "combined_dataset_fixed.csv"
x_data, y_data, class_mapping = load_and_preprocess_data(file_path)

print(f"\nDataset original: {len(x_data)} échantillons")
for class_id in np.unique(y_data):
    count = np.sum(y_data == class_id)
    print(f"  - {class_mapping[class_id]}: {count}")

# 2. SPLIT AVANT AUGMENTATION (CRUCIAL!)
print("\n🔑 SPLIT AVANT AUGMENTATION (anti-overfitting)")

X_train_orig, X_temp, y_train_orig, y_temp = train_test_split(
    x_data, y_data, test_size=0.4, random_state=RANDOM_SEED, stratify=y_data
)

X_val_orig, X_test_orig, y_val_orig, y_test_orig = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"\nSplit original (AVANT augmentation):")
print(f"  Train: {len(X_train_orig)} ({len(X_train_orig)/len(x_data)*100:.0f}%)")
print(f"  Val:   {len(X_val_orig)} ({len(X_val_orig)/len(x_data)*100:.0f}%)")
print(f"  Test:  {len(X_test_orig)} ({len(X_test_orig)/len(x_data)*100:.0f}%)")

# 3. Augmenter UNIQUEMENT le train set
print("\n🔄 Augmentation UNIQUEMENT du train set...")
X_train_aug, y_train_aug = augment_dataset(X_train_orig, y_train_orig, augmentation_factor=6)

print(f"  Train augmenté: {len(X_train_orig)} → {len(X_train_aug)} (×{len(X_train_aug)/len(X_train_orig):.1f})")
print(f"  Val reste:  {len(X_val_orig)} (données originales)")
print(f"  Test reste: {len(X_test_orig)} (données originales)")

# 4. Normaliser
print("\n🔄 Normalisation...")
X_train_normalized = normalize_data(X_train_aug)
X_val_normalized = normalize_data(X_val_orig)
X_test_normalized = normalize_data(X_test_orig)

# 5. Encoder
label_encoder = LabelEncoder()
label_encoder.fit(y_data)  # Fit sur toutes les classes

y_train = to_categorical(label_encoder.transform(y_train_aug), num_classes=3)
y_val = to_categorical(label_encoder.transform(y_val_orig), num_classes=3)
y_test = to_categorical(label_encoder.transform(y_test_orig), num_classes=3)

class_names = [class_mapping[c] for c in label_encoder.classes_]
print(f"✓ Classes: {class_names}")

print(f"\n📊 Données finales:")
print(f"  Train: {X_train_normalized.shape}")
print(f"  Val:   {X_val_normalized.shape}")
print(f"  Test:  {X_test_normalized.shape}")

# 6. Créer modèle
input_shape = (WINDOW_SIZE, NUM_FEATURES)
model = create_minimal_cnn(input_shape, 3)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*60)
print("ARCHITECTURE")
print("="*60)
model.summary()

# 7. Entraîner
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'stm32_proper_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n" + "="*60)
print("ENTRAÎNEMENT")
print("="*60)

history = model.fit(
    X_train_normalized, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_normalized, y_val),
    callbacks=callbacks,
    verbose=1
)

# 8. Évaluer
print("\n" + "="*60)
print("ÉVALUATION SUR DONNÉES NON-VUES")
print("="*60)

test_results = model.evaluate(X_test_normalized, y_test, verbose=0)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.2%}")

y_pred = model.predict(X_test_normalized, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nAccuracy par classe:")
for i, class_name in enumerate(class_names):
    indices = np.where(y_true_classes == i)[0]
    if len(indices) > 0:
        correct = np.sum(y_pred_classes[indices] == y_true_classes[indices])
        accuracy = correct / len(indices)
        print(f"  {class_name}: {accuracy:.2%} ({correct}/{len(indices)})")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nMatrice de confusion:")
print(cm)

# Analyser overfitting
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print("\n" + "="*60)
print("ANALYSE OVERFITTING")
print("="*60)
print(f"Train Accuracy: {train_acc:.2%}")
print(f"Val Accuracy:   {val_acc:.2%}")
print(f"Test Accuracy:  {test_results[1]:.2%}")
print(f"\nTrain Loss:     {train_loss:.4f}")
print(f"Val Loss:       {val_loss:.4f}")
print(f"Test Loss:      {test_results[0]:.4f}")

gap = abs(train_acc - val_acc)
if gap < 0.05:
    print(f"\n✓ Pas d'overfitting (gap: {gap:.1%})")
elif gap < 0.15:
    print(f"\n⚠️ Léger overfitting (gap: {gap:.1%})")
else:
    print(f"\n❌ Fort overfitting (gap: {gap:.1%})")

# 9. Conversion TFLite
print("\n" + "="*60)
print("CONVERSION TENSORFLOW LITE")
print("="*60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('stm32_proper.tflite', 'wb') as f:
    f.write(tflite_model)

size_float = len(tflite_model) / 1024

def representative_dataset():
    for i in range(min(100, len(X_train_normalized))):
        yield [X_train_normalized[i:i+1]]

converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
converter_q.representative_dataset = representative_dataset
converter_q.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_q.inference_input_type = tf.float32
converter_q.inference_output_type = tf.float32

tflite_quantized = converter_q.convert()

with open('stm32_proper_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized)

size_quantized = len(tflite_quantized) / 1024

print(f"\n1️⃣ Float32:   {size_float:.1f} KB")
print(f"2️⃣ Quantized: {size_quantized:.1f} KB")

estimated_total = size_quantized + 50
print("\n" + "="*60)
print("COMPATIBILITÉ STM32L476JG")
print("="*60)
print(f"\n📦 Flash:   {estimated_total:.1f} KB / 1024 KB")
print(f"   Utilisation: {estimated_total/1024*100:.1f}%")
print("   ✓✓✓ EXCELLENT!")

# Sauvegarder
import joblib
metadata = {
    'label_encoder': label_encoder,
    'class_mapping': class_mapping,
    'class_names': class_names,
    'window_size': WINDOW_SIZE,
    'num_features': NUM_FEATURES
}
joblib.dump(metadata, 'model_proper_metadata.pkl')

print("\n" + "="*60)
print("FICHIERS GÉNÉRÉS")
print("="*60)
print("✓ stm32_proper_model.h5")
print("✓ stm32_proper_quantized.tflite  ← UTILISEZ CELUI-CI")
print("✓ model_proper_metadata.pkl")

print("\n" + "="*60)
print("PROCHAINE ÉTAPE")
print("="*60)
print("1. Vérifiez Test Accuracy: doit être 50-80%")
print("2. Si bon, analysez dans CubeMX")
print("="*60)
