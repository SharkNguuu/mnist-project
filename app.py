import os
import pickle
import tempfile
import threading
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from data import get_splits
from final_models import _binarize
from forest import RandomForest
from mlp import MLP, train_mlp
from naive_bayes import BernoulliNB
import softmax as sm


MODEL_NAMES = {
    "MLP": "MLP (Mạng nơ-ron)",
    "RandomForest": "Random Forest",
    "Softmax": "Softmax Regression",
    "BernoulliNB": "Naive Bayes (Bernoulli)",
}

MODEL_CACHE = {}
MODEL_CACHE_DIR = Path(__file__).resolve().parent / ".model_cache"
MODEL_CACHE_VERSION = 1
MODEL_LOCK = threading.Lock()
DATASET: dict[str, np.ndarray] | None = None


def _get_dataset():
    global DATASET
    if DATASET is None:
        DATASET = get_splits()
    return DATASET


def _normalize_digit_image(image):
    """Convert a canvas image to MNIST-style white strokes on a black 28x28 image."""
    if image is None:
        raise ValueError("Vui lòng vẽ một chữ số trước khi dự đoán.")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("L")
    else:
        image = image.convert("L")

    pixels = np.asarray(image, dtype=np.float32)
    if pixels.size == 0:
        raise ValueError("Ảnh vẽ rỗng.")

    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.median(border) > 127:
        pixels = 255.0 - pixels

    peak = float(pixels.max())
    if peak < 8:
        raise ValueError("Không tìm thấy nét vẽ. Hãy vẽ chữ số đậm hơn.")

    foreground = pixels > max(8.0, peak * 0.08)
    ys, xs = np.where(foreground)
    if len(xs) == 0:
        raise ValueError("Không tìm thấy nét vẽ. Hãy vẽ chữ số đậm hơn.")

    # Crop only the digit, retaining grayscale antialiasing for smoother strokes.
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = Image.fromarray(np.clip(pixels[y0:y1, x0:x1], 0, 255).astype(np.uint8))

    scale = 20.0 / max(crop.size)
    resized_size = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    crop = crop.resize(resized_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - crop.width) // 2
    top = (28 - crop.height) // 2
    canvas.paste(crop, (left, top))

    # MNIST digits are centered by their intensity-weighted center of mass.
    centered = np.asarray(canvas, dtype=np.float32)
    mass = centered.sum()
    if mass > 0:
        yy, xx = np.indices(centered.shape)
        center_x = float((xx * centered).sum() / mass)
        center_y = float((yy * centered).sum() / mass)
        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))
        canvas = ImageOps.expand(canvas, border=28, fill=0).crop(
            (28 - shift_x, 28 - shift_y, 56 - shift_x, 56 - shift_y)
        )

    return np.asarray(canvas, dtype=np.float32).reshape(1, -1) / 255.0


def _get_model(model_name):
    with MODEL_LOCK:
        if model_name in MODEL_CACHE:
            return MODEL_CACHE[model_name]

        cache_path = MODEL_CACHE_DIR / f"{model_name}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as cache_file:
                cached = pickle.load(cache_file)
            if (
                cached.get("version") == MODEL_CACHE_VERSION
                and cached.get("model_name") == model_name
            ):
                model = cached["model"]
                MODEL_CACHE[model_name] = model
                return model

        dataset = _get_dataset()
        if model_name == "MLP":
            model = MLP(n_hidden=512, seed=0)
            train_mlp(
                model,
                dataset["X_train"],
                dataset["y_train"],
                dataset["X_val"],
                dataset["y_val"],
                lr=0.3,
                epochs=20,
                batch_size=128,
                seed=0,
                track=False,
            )
        elif model_name == "RandomForest":
            model = RandomForest(n_trees=61, max_depth=20, seed=1).fit(
                _binarize(dataset["X_train"]), dataset["y_train"]
            )
        elif model_name == "Softmax":
            model = sm.train(
                dataset["X_train"],
                dataset["y_train"],
                dataset["X_val"],
                dataset["y_val"],
                lr=0.1,
                epochs=30,
                batch_size=128,
                seed=0,
                track=False,
            )[:2]
        elif model_name == "BernoulliNB":
            model = BernoulliNB(alpha=1e-3, binarize=0.5).fit(
                dataset["X_train"], dataset["y_train"]
            )
        else:
            raise ValueError(f"Thuật toán không hỗ trợ: {model_name}")

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=MODEL_CACHE_DIR,
                prefix=f"{model_name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                pickle.dump(
                    {
                        "version": MODEL_CACHE_VERSION,
                        "model_name": model_name,
                        "model": model,
                    },
                    temp_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temp_path, cache_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        MODEL_CACHE[model_name] = model
        return model


def _class_probabilities(model_name, features):
    model = _get_model(model_name)
    if model_name == "MLP":
        logits = model.forward(features)[2]
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)
    if model_name == "RandomForest":
        return model.predict_proba(_binarize(features))
    if model_name == "Softmax":
        weights, bias = model
        return sm.softmax(features @ weights + bias)

    log_likelihood = model.joint_log_likelihood(features)
    log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
    probabilities = np.exp(log_likelihood)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    full_probabilities = np.zeros((len(features), 10), dtype=np.float64)
    full_probabilities[:, model.classes_] = probabilities
    return full_probabilities


def predict_all_models(sketch_data):
    """Predict with every model and return the best guess and confidence table."""
    if isinstance(sketch_data, dict):
        image = sketch_data.get("composite")
    else:
        image = sketch_data

    if image is None:
        return "Vui lòng vẽ một chữ số (0-9) trước.", []

    try:
        features = _normalize_digit_image(image)
        rows = []
        for model_name, label in MODEL_NAMES.items():
            probabilities = _class_probabilities(model_name, features)[0]
            top_digit = int(np.argmax(probabilities))
            confidence = float(probabilities[top_digit]) * 100.0
            rows.append([label, top_digit, f"{confidence:.1f}%"])
    except (TypeError, ValueError, OSError) as exc:
        return f"Không thể xử lý ảnh: {exc}", []
    except Exception as exc:
        return f"Lỗi khi chạy mô hình: {exc}", []

    model_predictions = [int(row[1]) for row in rows]
    counts = np.bincount(model_predictions, minlength=10)
    consensus = int(np.flatnonzero(counts == counts.max())[0])
    mlp_prediction = model_predictions[0]
    agreement = int(counts[consensus])
    summary = (
        f"MLP dự đoán **{mlp_prediction}**. "
        f"{agreement}/4 mô hình cùng dự đoán **{consensus}**. "
        "Confidence của các mô hình không được hiệu chuẩn giống nhau, "
        "nên không nên so trực tiếp để chọn mô hình thắng."
    )
    return summary, rows


with gr.Blocks() as demo:
    gr.Markdown("# Nhận dạng chữ số viết tay")
    gr.Markdown(
        "Vẽ một chữ số từ 0 đến 9. Hệ thống sẽ chuẩn hóa ảnh theo MNIST "
        "và so sánh confidence của cả bốn mô hình."
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            sketch = gr.Sketchpad(
                label="Bảng vẽ chữ số",
                width=420,
                height=420,
                type="numpy",
                image_mode="RGBA",
                brush=gr.Brush(
                    default_size=18,
                    default_color="#111827",
                    color_mode="fixed",
                ),
            )
            with gr.Row():
                predict_button = gr.Button("Dự đoán", variant="primary")
                clear_button = gr.ClearButton(
                    [sketch],
                    value="Xóa bảng vẽ",
                    size="lg",
                )

        with gr.Column(scale=1):
            gr.Markdown("### Kết quả")
            summary = gr.Markdown("Confidence sẽ hiển thị sau khi dự đoán.")
            results = gr.Dataframe(
                headers=["Mô hình", "Dự đoán", "Confidence"],
                datatype=["str", "number", "str"],
                interactive=False,
                label="So sánh mô hình",
            )
            gr.Markdown(
                "- Confidence là mức độ tin cậy của mô hình, không đảm bảo chữ số đúng."
            )

    predict_button.click(
        fn=predict_all_models,
        inputs=sketch,
        outputs=[summary, results],
    )
    clear_button.click(
        fn=lambda: ("Confidence sẽ hiển thị sau khi dự đoán.", []),
        outputs=[summary, results],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(),
    )
