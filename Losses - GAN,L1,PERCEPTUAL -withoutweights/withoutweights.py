import os
import glob
import time
import hashlib
import datetime
import copy

import cv2
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Dropout,
    Input, Flatten, Concatenate, LeakyReLU, Dense
)
from tensorflow.keras.applications.vgg16 import VGG16

# =========================
# PATHS / SETTINGS
# =========================
PATIENT_ROOT = "/storage/scratch2/dl-abdominal-mri-gan/new"
ROOT = "/storage/scratch2/dl-abdominal-mri-gan/new"
SAVE_DIR = "/storage/scratch2/dl-abdominal-mri-gan/chunk_epoch_training/withoutweights"
os.makedirs(SAVE_DIR, exist_ok=True)

VALID_FILES_CACHE = os.path.join(SAVE_DIR, "valid_dicom_files.txt")
SPLIT_CACHE = os.path.join(SAVE_DIR, "train_test_split.npz")
MOTION_CACHE_DIR = os.path.join(SAVE_DIR, "motion_cache")
os.makedirs(MOTION_CACHE_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(SAVE_DIR, "tf_gan_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")

LOSS_CSV_PATH = os.path.join(SAVE_DIR, "gan_loss_history.csv")
LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "g_loss_vs_d_loss.png")
SAMPLE_DIR = os.path.join(SAVE_DIR, "sample_predictions")
os.makedirs(SAMPLE_DIR, exist_ok=True)

IMG_SIZE = 512
TRAIN_RATIO = 0.80
BUFFER_SIZE = 200
BATCH_SIZE = 2
EPOCHS = 50
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

LAMBDA1 = 0.008
LAMBDA2 = 0.01
LAMBDA3 = 0.006

# =========================
# DICOM DISCOVERY / SPLIT
# =========================
def is_valid_dicom(filepath: str) -> bool:
    try:
        ds = pydicom.dcmread(filepath, force=True)
        _ = ds.pixel_array
        return True
    except Exception:
        return False


def get_all_dicom_files(root: str, cache_file: str = None):
    if cache_file is not None and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"Loaded cached valid DICOM list: {len(files)} files")
        return files

    print("Scanning for valid DICOM files first run only ...")
    files = glob.glob(os.path.join(root, "**", "*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files = [f for f in files if is_valid_dicom(f)]
    files.sort()

    if cache_file is not None:
        with open(cache_file, "w") as f:
            for p in files:
                f.write(p + "\n")

    return files


def make_train_test_split(all_files, train_ratio=TRAIN_RATIO, seed=42):
    if os.path.exists(SPLIT_CACHE):
        data = np.load(SPLIT_CACHE)
        train_idx = data["train_idx"].tolist()
        test_idx = data["test_idx"].tolist()
        print(f"Loaded existing split -- train: {len(train_idx)}, test: {len(test_idx)}")
    else:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        train_idx = sorted(indices[:n_train].tolist())
        test_idx = sorted(indices[n_train:].tolist())

        np.savez(SPLIT_CACHE, train_idx=train_idx, test_idx=test_idx)
        print(f"Created new split -- train: {len(train_idx)}, test: {len(test_idx)}")

    train_files = [all_files[i] for i in train_idx]
    test_files = [all_files[i] for i in test_idx]
    return train_files, test_files, train_idx, test_idx


all_files = get_all_dicom_files(ROOT, cache_file=VALID_FILES_CACHE)
print(f"Total dicoms: {len(all_files)}")

train_files, test_files, train_idx, test_idx = make_train_test_split(
    all_files,
    train_ratio=TRAIN_RATIO,
    seed=SEED
)

# =========================
# DICOM PREPROCESSING + MOTION ARTIFACT
# =========================
def normalize_dicom_pixels(arr):
    arr = arr.astype(np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        arr = arr[..., 0]

    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-7)
    return arr.astype(np.float32)


def read_dicom_clean(path):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array
    arr = normalize_dicom_pixels(arr)
    arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    arr = arr[..., np.newaxis]
    return arr.astype(np.float32)


def add_motion_artifact(clean, seed=None):
    rng = np.random.default_rng(seed)

    img = np.squeeze(clean).astype(np.float32)
    h, w = img.shape

    k_clean = np.fft.fftshift(np.fft.fft2(img))
    k_motion = k_clean.copy()

    n_events = int(rng.integers(2, 5))

    for _ in range(n_events):
        shift_y = int(rng.integers(-12, 13))
        shift_x = int(rng.integers(-12, 13))

        moved = np.roll(img, shift=(shift_y, shift_x), axis=(0, 1))
        k_moved = np.fft.fftshift(np.fft.fft2(moved))

        band_center = int(rng.integers(0, h))
        band_width = int(rng.integers(max(2, h // 80), max(3, h // 25)))

        y0 = max(0, band_center - band_width)
        y1 = min(h, band_center + band_width)

        k_motion[y0:y1, :] = k_moved[y0:y1, :]

    corrupted = np.abs(np.fft.ifft2(np.fft.ifftshift(k_motion)))
    corrupted = corrupted - corrupted.min()
    corrupted = corrupted / (corrupted.max() + 1e-7)

    return corrupted[..., np.newaxis].astype(np.float32)


def cache_name(path):
    return hashlib.md5(path.encode("utf-8")).hexdigest() + ".npy"


def load_pair_numpy(path_bytes):
    path = path_bytes.decode("utf-8") if isinstance(path_bytes, bytes) else path_bytes.numpy().decode("utf-8")

    clean = read_dicom_clean(path)

    cache_path = os.path.join(MOTION_CACHE_DIR, cache_name(path))

    if os.path.exists(cache_path):
        motion = np.load(cache_path).astype(np.float32)
    else:
        seed = int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)
        motion = add_motion_artifact(clean, seed=seed)
        np.save(cache_path, motion)

    input_image = motion * 2.0 - 1.0
    real_image = clean * 2.0 - 1.0

    return input_image.astype(np.float32), real_image.astype(np.float32)


def load_pair_tf(path):
    input_image, real_image = tf.numpy_function(
        func=load_pair_numpy,
        inp=[path],
        Tout=[tf.float32, tf.float32]
    )

    input_image.set_shape([IMG_SIZE, IMG_SIZE, 1])
    real_image.set_shape([IMG_SIZE, IMG_SIZE, 1])

    return input_image, real_image


# =========================
# DATASET PIPELINE WITH REPEAT FIX
# =========================
def make_dataset(file_list, batch_size=BATCH_SIZE, shuffle=True, repeat=False):
    ds = tf.data.Dataset.from_tensor_slices(file_list)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(BUFFER_SIZE, len(file_list)),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(load_pair_tf, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(AUTOTUNE)

    return ds


train_dataset = make_dataset(
    train_files,
    batch_size=BATCH_SIZE,
    shuffle=True,
    repeat=True
)

test_dataset = make_dataset(
    test_files,
    batch_size=BATCH_SIZE,
    shuffle=False,
    repeat=False
)

test_dataset_shuffle = make_dataset(
    test_files,
    batch_size=BATCH_SIZE,
    shuffle=True,
    repeat=True
)

steps_per_epoch = max(1, len(train_files) // BATCH_SIZE)
validation_steps = max(1, len(test_files) // BATCH_SIZE)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# =========================
# GENERATOR
# =========================
n_f = 32


def conv_block(x, n_f, strides_x, strides_y):
    x = Conv2D(
        filters=n_f,
        kernel_size=(3, 3),
        strides=(strides_x, strides_y),
        kernel_initializer="he_normal",
        padding="same"
    )(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
    x = LeakyReLU(alpha=0.3)(x)
    return x


def m_block(x):
    x2 = conv_block(x, n_f, 1, 1)
    x2 = conv_block(x2, n_f, 1, 1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x3 = conv_block(pool2, n_f, 1, 1)
    x3 = conv_block(x3, n_f, 1, 1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)

    x4 = conv_block(pool3, n_f, 1, 1)
    x4 = conv_block(x4, n_f, 1, 1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)

    x5 = conv_block(pool4, n_f, 1, 1)
    x5 = conv_block(x5, n_f, 1, 1)
    pool5 = MaxPooling2D(pool_size=(2, 2))(x5)

    x6 = conv_block(pool5, n_f, 1, 1)
    x6 = conv_block(x6, n_f, 1, 1)

    up7 = UpSampling2D(size=(2, 2))(x6)
    up7 = conv_block(up7, n_f, 1, 1)
    merge7 = Concatenate(axis=-1)([x5, up7])
    x7 = conv_block(merge7, n_f, 1, 1)
    x7 = conv_block(x7, n_f, 1, 1)

    up8 = UpSampling2D(size=(2, 2))(x7)
    up8 = conv_block(up8, n_f, 1, 1)
    merge8 = Concatenate(axis=-1)([x4, up8])
    x8 = conv_block(merge8, n_f, 1, 1)
    x8 = conv_block(x8, n_f, 1, 1)

    up9 = UpSampling2D(size=(2, 2))(x8)
    up9 = conv_block(up9, n_f, 1, 1)
    merge9 = Concatenate(axis=-1)([x3, up9])
    x9 = conv_block(merge9, n_f, 1, 1)
    x9 = conv_block(x9, n_f, 1, 1)

    up10 = UpSampling2D(size=(2, 2))(x9)
    up10 = conv_block(up10, n_f, 1, 1)
    merge10 = Concatenate(axis=-1)([x2, up10])
    x10 = conv_block(merge10, n_f, 1, 1)
    x10 = conv_block(x10, n_f, 1, 1)

    return x10


def dense_block(x, n_layer):
    list_feat = [x]

    for _ in range(n_layer):
        x = m_block(x)
        list_feat.append(x)
        x = Concatenate(axis=-1)(copy.copy(list_feat))

    return x


inpt = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x_b = dense_block(inpt, 3)
x_b = conv_block(x_b, 1, 1, 1)
x_out = tf.keras.layers.Subtract()([inpt, x_b])

generator = Model(inputs=inpt, outputs=x_out, name="generator")
# generator.summary()

# =========================
# DISCRIMINATOR
# =========================
def conv_block_dis(x, n_f_dis, strides_x, strides_y):
    x = Conv2D(
        filters=n_f_dis,
        kernel_size=(3, 3),
        strides=(strides_x, strides_y),
        kernel_initializer="he_normal",
        padding="same"
    )(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
    x = LeakyReLU(alpha=0.3)(x)
    return x


image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

x = conv_block_dis(image_input, n_f, 4, 4)
x = conv_block_dis(x, n_f, 4, 4)
x = conv_block_dis(x, n_f, 4, 4)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(1)(x)

discriminator = Model(inputs=image_input, outputs=x, name="discriminator")
# discriminator.summary()

# =========================
# LOSSES
# =========================
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

lossModel = VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
lossModel.trainable = False

vgg_layer_15 = Model(lossModel.inputs, lossModel.layers[15].output)
vgg_layer_16 = Model(lossModel.inputs, lossModel.layers[16].output)
vgg_layer_17 = Model(lossModel.inputs, lossModel.layers[17].output)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    gen_output_scale = (gen_output + 1.0) * 0.5 * 255.0
    target_scale = (target + 1.0) * 0.5 * 255.0

    vgg_input = tf.concat([gen_output_scale, gen_output_scale, gen_output_scale], axis=-1)
    target_vgg_input = tf.concat([target_scale, target_scale, target_scale], axis=-1)

    vgg_input = tf.keras.applications.vgg16.preprocess_input(vgg_input)
    target_vgg_input = tf.keras.applications.vgg16.preprocess_input(target_vgg_input)

    percep_loss_15 = tf.keras.losses.MSE(
        vgg_layer_15(target_vgg_input),
        vgg_layer_15(vgg_input)
    )
    percep_loss_15 = K.mean(percep_loss_15, axis=[0, 1, 2])

    percep_loss_16 = tf.keras.losses.MSE(
        vgg_layer_16(target_vgg_input),
        vgg_layer_16(vgg_input)
    )
    percep_loss_16 = K.mean(percep_loss_16, axis=[0, 1, 2])

    percep_loss_17 = tf.keras.losses.MSE(
        vgg_layer_17(target_vgg_input),
        vgg_layer_17(vgg_input)
    )
    percep_loss_17 = K.mean(percep_loss_17, axis=[0, 1, 2])

    percep_loss = percep_loss_15 + percep_loss_16 + percep_loss_17
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = (
        LAMBDA1 * gan_loss +
        LAMBDA2 * l1_loss +
        LAMBDA3 * percep_loss
    )

    return total_gen_loss, gan_loss, l1_loss, percep_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss


# =========================
# OPTIMIZERS / CHECKPOINTS
# =========================
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)

latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)

if latest:
    checkpoint.restore(latest)
    print(f"Restored checkpoint: {latest}")

# =========================
# RESUME FROM OLD LOSS CSV
# =========================
if os.path.exists(LOSS_CSV_PATH):
    old_history_df = pd.read_csv(LOSS_CSV_PATH)

    if len(old_history_df) > 0 and "epoch" in old_history_df.columns:
        initial_epoch = int(old_history_df["epoch"].max())
        print(f"Previous training found until epoch {initial_epoch}")
        print(f"Next training will start from epoch {initial_epoch + 1}")
    else:
        initial_epoch = 0
        old_history_df = pd.DataFrame()
else:
    initial_epoch = 0
    old_history_df = pd.DataFrame()
# =========================
# TRAINING STEP
# =========================
@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator(target, training=True)
        disc_generated_output = discriminator(gen_output, training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss, per_loss = generator_loss(
            disc_generated_output,
            gen_output,
            target
        )

        disc_loss = discriminator_loss(
            disc_real_output,
            disc_generated_output
        )

    generator_gradients = gen_tape.gradient(
        gen_total_loss,
        generator.trainable_variables
    )

    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return gen_total_loss, gen_gan_loss, gen_l1_loss, per_loss, disc_loss


@tf.function
def val_step(input_image, target, epoch=None):
    gen_output = generator(input_image, training=False)

    disc_real_output = discriminator(target, training=False)
    disc_generated_output = discriminator(gen_output, training=False)

    gen_total_loss, gen_gan_loss, gen_l1_loss, per_loss = generator_loss(
        disc_generated_output,
        gen_output,
        target
    )

    disc_loss = discriminator_loss(
        disc_real_output,
        disc_generated_output
    )

    return gen_total_loss, gen_gan_loss, gen_l1_loss, per_loss, disc_loss


# =========================
# PLOT G LOSS VS D LOSS ONLY
# =========================
def plot_g_loss_vs_d_loss(history):
    history_df = pd.DataFrame(history)

    plt.figure(figsize=(8, 6))

    plt.plot(history_df["epoch"], history_df["gen_total_loss"], label="G loss")
    plt.plot(history_df["epoch"], history_df["disc_loss"], label="D loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss vs Discriminator Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()

    print(f"Saved graph to: {LOSS_PLOT_PATH}")

# =========================
# SAMPLE IMAGE DISPLAY
# =========================
def generate_images(model, test_input, target, epoch=None):
    prediction = model(test_input, training=False)

    plt.figure(figsize=(15, 5))

    display_list = [test_input[0], target[0], prediction[0]]
    title = ["Input Motion DICOM", "Ground Truth", "Predicted"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i][:, :, 0] * 0.5 + 0.5, cmap="gray")
        plt.axis("off")

    if epoch is not None:
        sample_path = os.path.join(SAMPLE_DIR, f"sample_epoch_{epoch}.png")
        plt.savefig(sample_path, bbox_inches="tight", dpi=150)

    plt.close()


# =========================
# FIT
# =========================
def mean_losses_over_dataset(dataset, step_function, num_steps, epoch=None):
    sums = [0.0, 0.0, 0.0, 0.0, 0.0]
    count = 0

    for input_image, target in dataset.take(num_steps):
        if epoch is None:
            losses = step_function(input_image, target, None)
        else:
            losses = step_function(input_image, target, epoch)

        for i, loss in enumerate(losses):
            sums[i] += float(loss.numpy())

        count += 1

    if count == 0:
        raise ValueError("Dataset is empty. Check your ROOT path and DICOM files.")

    return [s / count for s in sums]


def fit(train_ds, test_ds, epochs):
    if os.path.exists(LOSS_CSV_PATH):
        history_df = pd.read_csv(LOSS_CSV_PATH)
        history = history_df.to_dict(orient="list")
    else:
        history = {
            "epoch": [],
            "gen_total_loss": [],
            "gen_gan_loss": [],
            "gen_l1_loss": [],
            "per_loss": [],
            "disc_loss": [],
            "gen_total_loss_val": [],
            "gen_gan_loss_val": [],
            "gen_l1_loss_val": [],
            "per_loss_val": [],
            "disc_loss_val": []
        }

    best_val_loss = np.inf
    if len(history["gen_total_loss_val"]) > 0:
        best_val_loss = min(history["gen_total_loss_val"])

    patience_count = 0
    patience_max = 30

    for epoch in range(initial_epoch, epochs):
        start = time.time()
        epoch_number = epoch + 1

        print(f"\nEpoch {epoch_number}/{epochs}")

        for example_input, example_target in test_dataset_shuffle.take(1):
            generate_images(generator, example_input, example_target, epoch=epoch_number)

        train_losses = mean_losses_over_dataset(
            train_ds,
            train_step,
            steps_per_epoch,
            epoch
        )

        val_losses = mean_losses_over_dataset(
            test_ds,
            val_step,
            validation_steps,
            None
        )

        gen_total_loss, gen_gan_loss, gen_l1_loss, per_loss, disc_loss = train_losses
        gen_total_loss_val, gen_gan_loss_val, gen_l1_loss_val, per_loss_val, disc_loss_val = val_losses

        print(
            f"G loss: {gen_total_loss:.5f}, "
            f"D loss: {disc_loss:.5f}, "
            f"Val G loss: {gen_total_loss_val:.5f}, "
            f"Val D loss: {disc_loss_val:.5f}"
        )

        history["epoch"].append(epoch_number)

        history["gen_total_loss"].append(gen_total_loss)
        history["gen_gan_loss"].append(gen_gan_loss)
        history["gen_l1_loss"].append(gen_l1_loss)
        history["per_loss"].append(per_loss)
        history["disc_loss"].append(disc_loss)

        history["gen_total_loss_val"].append(gen_total_loss_val)
        history["gen_gan_loss_val"].append(gen_gan_loss_val)
        history["gen_l1_loss_val"].append(gen_l1_loss_val)
        history["per_loss_val"].append(per_loss_val)
        history["disc_loss_val"].append(disc_loss_val)

        history_df = pd.DataFrame(history)
        history_df = history_df.drop_duplicates(subset=["epoch"], keep="last")
        history_df = history_df.sort_values("epoch")

        history_df.to_csv(LOSS_CSV_PATH, index=False)
        plot_g_loss_vs_d_loss(history_df)

        checkpoint.save(file_prefix=CHECKPOINT_PREFIX)
        print("Saved checkpoint for resume.")

        if gen_total_loss_val < best_val_loss:
            best_val_loss = gen_total_loss_val
            patience_count = 0
            print("Validation improved.")
        else:
            patience_count += 1
            print(f"No validation improvement. Patience {patience_count}/{patience_max}")

            if patience_count >= patience_max:
                print("Early stopping.")
                break

        print(f"Time taken: {time.time() - start:.2f} sec")

    checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

    history_df = pd.DataFrame(history)
    history_df = history_df.drop_duplicates(subset=["epoch"], keep="last")
    history_df = history_df.sort_values("epoch")
    history_df.to_csv(LOSS_CSV_PATH, index=False)
    plot_g_loss_vs_d_loss(history_df)

    return history_df
# =========================
# START TRAINING
# =========================
history = fit(train_dataset, test_dataset, EPOCHS)

# =========================
# FULL EVALUATION + METRIC PLOTS
# =========================
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

METRIC_PLOT_PATH = os.path.join(SAVE_DIR, "evaluation_metrics.png")
METRIC_CSV_PATH = os.path.join(SAVE_DIR, "evaluation_metrics.csv")


def evaluate_metrics(dataset):
    SS = []
    MSE = []
    PSNR = []
    sample_id = []
    count = 0

    for input_image, target in dataset:
        prediction = generator(input_image, training=False).numpy()
        target_np = target.numpy()

        for i in range(prediction.shape[0]):
            target_img = target_np[i, :, :, 0]
            pred_img = prediction[i, :, :, 0]

            SS.append(ssim(target_img, pred_img, data_range=2.0))
            MSE.append(mse(target_img, pred_img))
            PSNR.append(psnr(target_img, pred_img, data_range=2.0))

            sample_id.append(count)
            count += 1

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"SSIM Mean : {np.mean(SS):.5f}")
    print(f"SSIM Std  : {np.std(SS):.5f}")
    print(f"MSE Mean  : {np.mean(MSE):.5f}")
    print(f"MSE Std   : {np.std(MSE):.5f}")
    print(f"PSNR Mean : {np.mean(PSNR):.5f}")
    print(f"PSNR Std  : {np.std(PSNR):.5f}")

    metrics_df = pd.DataFrame({
        "Sample": sample_id,
        "SSIM": SS,
        "MSE": MSE,
        "PSNR": PSNR
    })

    metrics_df.to_csv(METRIC_CSV_PATH, index=False)
    print(f"Saved metrics CSV to: {METRIC_CSV_PATH}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(SS)
    plt.title("SSIM")
    plt.xlabel("Sample")
    plt.ylabel("SSIM")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(MSE)
    plt.title("MSE")
    plt.xlabel("Sample")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(PSNR)
    plt.title("PSNR")
    plt.xlabel("Sample")
    plt.ylabel("PSNR")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(METRIC_PLOT_PATH, dpi=150)
    plt.close()

    print(f"Saved metric plots to: {METRIC_PLOT_PATH}")

    return SS, MSE, PSNR


SS, MSE, PSNR = evaluate_metrics(test_dataset)