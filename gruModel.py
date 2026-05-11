# ============================================================
# GRU Seq2Seq Data Cleaning Model (FULL RESUMABLE KAGGLE PIPELINE)
# ============================================================

import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Concatenate
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "gru_cleaning_model.h5"
max_vocab_size = 10000
max_len = 20
embedding_dim = 128
gru_units = 256

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv('/kaggle/input/datasets/abdulbasit1287/cleanlinesslearningmodelds/cleanlinessLearningModelDS.csv')

input_texts = df['raw_record'].astype(str).tolist()
target_texts = df['cleaned_record'].astype(str).tolist()

target_texts = ['<start> ' + t + ' <end>' for t in target_texts]

# ============================================================
# TOKENIZATION (ALWAYS REQUIRED)
# ============================================================

tokenizer = Tokenizer(num_words=max_vocab_size, filters='')
tokenizer.fit_on_texts(input_texts + target_texts)

input_seq = tokenizer.texts_to_sequences(input_texts)
target_seq = tokenizer.texts_to_sequences(target_texts)

input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
target_seq = pad_sequences(target_seq, maxlen=max_len, padding='post')

vocab_size = min(max_vocab_size, len(tokenizer.word_index) + 1)

decoder_input = target_seq[:, :-1]
decoder_target = target_seq[:, 1:]

# ============================================================
# TRAIN SPLIT
# ============================================================

X_train, X_val, y_in_train, y_in_val, y_out_train, y_out_val = train_test_split(
    input_seq,
    decoder_input,
    decoder_target,
    test_size=0.2,
    random_state=42
)

# ============================================================
# MODEL LOAD OR BUILD
# ============================================================

if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Building new model...")

    # ---------------- ENCODER ----------------
    encoder_inputs = Input(shape=(max_len,))
    enc_emb = Embedding(vocab_size, embedding_dim, mask_zero=False)(encoder_inputs)

    encoder_outputs, encoder_state = GRU(
        gru_units,
        return_sequences=True,
        return_state=True
    )(enc_emb)

    # ---------------- DECODER ----------------
    decoder_inputs = Input(shape=(max_len - 1,))

    dec_emb_layer = Embedding(vocab_size, embedding_dim, mask_zero=False)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_gru = GRU(
        gru_units,
        return_sequences=True,
        return_state=True
    )

    decoder_outputs, _ = decoder_gru(dec_emb, initial_state=encoder_state)

    # ---------------- ATTENTION ----------------
    attention_layer = AdditiveAttention()
    attention_output = attention_layer([decoder_outputs, encoder_outputs])

    concat = Concatenate(axis=-1)([decoder_outputs, attention_output])

    # ---------------- OUTPUT ----------------
    outputs = Dense(vocab_size, activation='softmax')(concat)

    model = Model([encoder_inputs, decoder_inputs], outputs)

# ============================================================
# COMPILE (ALWAYS SAFE)
# ============================================================

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ============================================================
# CALLBACKS (RESUME + BEST MODEL SAVE)
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# ============================================================
# TRAINING (RESUMABLE)
# ============================================================

history = model.fit(
    [X_train, y_in_train],
    y_out_train,
    validation_data=([X_val, y_in_val], y_out_val),
    batch_size=32,
    epochs=30,
    callbacks=[early_stop, checkpoint]
)

# ============================================================
# FINAL SAVE
# ============================================================

model.save(MODEL_PATH)
print("Model saved successfully!")
