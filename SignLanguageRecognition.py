#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from collections import deque


# In[2]:


TRAIN_DIR = r"Z:\Shikhar\ASL\Sign-Language-Recognition\asl_alphabet_train\asl_alphabet_train"
TEST_DIR = r"Z:\Shikhar\ASL\Sign-Language-Recognition\asl_alphabet_test\asl_alphabet_test"

IMG_SIZE   = 64
BATCH_SIZE = 32


# In[3]:


datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    validation_split=0.2   
)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

labels = {v:k for k,v in train_generator.class_indices.items()}
print("Labels:", labels)


# In[4]:


model = models.Sequential([

    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),


    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),


    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.4),


    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# In[5]:


from tensorflow.keras.models import load_model

model = load_model("SLR_final.h5")

print("Model loaded successfully!")


# In[7]:


cb = [
    callbacks.ModelCheckpoint("SLR_final.h5", save_best_only=True, monitor='val_accuracy', mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=cb
)

print("Training finished. Best model saved as SLR_final.h5")


# In[ ]:


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()


# In[ ]:


TEST_DIR = r"Z:\Shikhar\ASL\Sign-Language-Recognition\asl_alphabet_test\asl_alphabet_test"
print(os.listdir(TEST_DIR)[:10])


# In[ ]:


test_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3]
)

test_gen = test_aug.flow_from_directory(
    TEST_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(test_gen.classes, y_pred)
class_labels = list(test_gen.class_indices.keys())

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(14, 12))
disp.plot(cmap="Blues", xticks_rotation=90, ax=ax)
plt.title("Confusion Matrix - Augmented Test Set")
plt.show()

acc = accuracy_score(test_gen.classes, y_pred)
print(f"Test Accuracy (Augmented): {acc*100:.2f}%")


# In[ ]:


pred_history = deque(maxlen=10)

cap = cv2.VideoCapture(0)
print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    roi = frame[y1:y2, x1:x2]

    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi.astype("float32")/255.0
    roi = np.expand_dims(roi, axis=0)

    pred = model.predict(roi, verbose=0)[0]
    pred_history.append(pred)
    avg_pred = np.mean(pred_history, axis=0)

    class_id = np.argmax(avg_pred)
    class_name = labels[class_id]
    confidence = avg_pred[class_id] * 100

    text = f"{class_name}: {confidence:.2f}%"
    cv2.putText(frame, text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

