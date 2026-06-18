import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# Keras 3 imports - changed from tensorflow.keras to keras
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ===== CONFIG =====
DATASET_DIR = 'dataset'
MODEL_PATH = 'models/weapon_detector.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

print("\n" + "="*50)
print("  WEAPON DETECTION - MODEL TRAINING")
print("="*50)

# Step 1: Check dataset
weapon_dir = os.path.join(DATASET_DIR, 'weapon')
no_weapon_dir = os.path.join(DATASET_DIR, 'no_weapon')

w_count = len([f for f in os.listdir(weapon_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
nw_count = len([f for f in os.listdir(no_weapon_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
print(f"\nWeapon images: {w_count}")
print(f"Non-weapon images: {nw_count}")

if w_count == 0 or nw_count == 0:
    print("\nERROR: Add images to dataset/weapon/ and dataset/no_weapon/")
    exit()

# Step 2: Load datasets
print("\nCreating datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Normalize and augment
normalization = tf.keras.layers.Rescaling(1.0/255.0)
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.2, 0.2),
])

train_ds = train_ds.map(lambda x, y: (normalization(augmentation(x)), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

print(f"Datasets ready!")

# Step 3: Build model
print("\nBuilding model (MobileNetV2)...")

base_model = MobileNetV2(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy', metrics=['accuracy'])

print(f"Total params: {model.count_params():,}")

# Step 4: Train Phase 1
os.makedirs('models', exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

print("\nPhase 1: Training classification head...")
history1 = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

# Step 5: Fine-tune
print("\nPhase 2: Fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10),
              loss='binary_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks)

# Save model
model.save(MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# Plot results
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss_h = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(acc, label='Train Acc')
ax1.plot(val_acc, label='Val Acc')
ax1.set_title('Accuracy')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

ax2.plot(loss_h, label='Train Loss')
ax2.plot(val_loss, label='Val Loss')
ax2.set_title('Loss')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_plots.png', dpi=150)
plt.close()

print("Training plots saved to: models/training_plots.png")
print("\nDONE! Now run: python manage.py runserver")