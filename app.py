import os
import json
import hashlib
import numpy as np
import pandas as pd
import keras
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="lasseu/eurosat-cnn",
    filename="RGB_batchnorm_3conv_full_best.keras"
)
model = keras.models.load_model(model_path)

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

CLASS_COLORS = {
    "AnnualCrop":           (255, 200, 0),
    "Forest":               (0, 100, 0),
    "HerbaceousVegetation": (0, 200, 0),
    "Highway":              (128, 128, 128),
    "Industrial":           (200, 0, 200),
    "Pasture":              (180, 255, 100),
    "PermanentCrop":        (255, 100, 0),
    "Residential":          (255, 255, 255),
    "River":                (0, 0, 255),
    "SeaLake":              (0, 200, 255),
}

EXAMPLE_PATHS = [
    # 10 mixed canvases
    *[f"demo/canvases/Mixed_{i}_{10}x{10}.jpg" for i in range(1, 11)],
    # Single-class canvases
    "demo/canvases/Forest_10x10.jpg",
    "demo/canvases/SeaLake_10x10.jpg",
    "demo/canvases/HerbaceousVegetation_10x10.jpg",
    "demo/canvases/Industrial_10x10.jpg",
    "demo/canvases/Highway_10x10.jpg",
    "demo/canvases/Residential_10x10.jpg",
    "demo/canvases/River_10x10.jpg",
    "demo/canvases/Pasture_10x10.jpg",
    "demo/canvases/PermanentCrop_10x10.jpg",
    "demo/canvases/AnnualCrop_10x10.jpg",
]

EXAMPLE_LABELS = [
    os.path.splitext(os.path.basename(p))[0] for p in EXAMPLE_PATHS
]


def _image_hash(pil_image):
    """MD5 hash of raw pixel data — used to match input images to canvas metadata."""
    arr = np.array(pil_image.convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def load_canvas_metadata(path):
    """Loads tile labels from the metadata JSON saved alongside a canvas."""
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


# Precompute hash → metadata for all canvases at startup.
CANVAS_META_BY_HASH = {}
for _path in EXAMPLE_PATHS:
    _meta = load_canvas_metadata(_path)
    if _meta and os.path.exists(_path):
        _h = _image_hash(Image.open(_path))
        CANVAS_META_BY_HASH[_h] = _meta


def get_canvas_meta(pil_image):
    """Returns metadata for the image if it matches a known canvas, else None."""
    return CANVAS_META_BY_HASH.get(_image_hash(pil_image))

def blend(original_arr, color_arr, opacity):
    """Blends color layer over original image at the given opacity (0-100)."""
    original_pil = Image.fromarray(original_arr).convert("RGBA")
    color_pil    = Image.fromarray(color_arr).convert("RGBA")
    return Image.blend(original_pil, color_pil, alpha=opacity / 100).convert("RGB")
 
 
WRONG_COLOR   = (255, 0, 0)    # red border for incorrect tiles
BORDER        = 3              # border thickness in pixels


def draw_correctness_borders(color_layer, predictions_grid, tile_labels, grid_cols):
    """
    Draws a red border on incorrect tiles only.
    """
    result = color_layer.copy()
    n_rows = len(predictions_grid)
    n_cols = len(predictions_grid[0])
    b = BORDER
    for row in range(n_rows):
        for col in range(n_cols):
            tile_idx = row * grid_cols + col
            if tile_idx >= len(tile_labels):
                continue
            true_class    = tile_labels[tile_idx]
            pred_class, _, _ = predictions_grid[row][col]
            if pred_class != true_class:
                y, x = row * 64, col * 64
                result[y:y+b,       x:x+64] = WRONG_COLOR
                result[y+64-b:y+64, x:x+64] = WRONG_COLOR
                result[y:y+64, x:x+b]       = WRONG_COLOR
                result[y:y+64, x+64-b:x+64] = WRONG_COLOR
    return result


def classify_image(image, opacity):
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]
    tile = 64

    h_crop = (h // tile) * tile
    w_crop = (w // tile) * tile

    if h_crop == 0 or w_crop == 0:
        raise gr.Error("Image too small — upload an image at least 64x64 pixels.")

    img = img[:h_crop, :w_crop]
    color_layer = np.zeros((h_crop, w_crop, 3), dtype=np.uint8)

    class_counts      = {cls: 0  for cls in CLASSES}
    class_confidences = {cls: [] for cls in CLASSES}
    last_probs        = None
    last_class        = None
    total_tiles       = 0

    n_rows = h_crop // tile
    n_cols = w_crop // tile
    predictions_grid = [[None] * n_cols for _ in range(n_rows)]

    # Look up metadata by pixel hash — works regardless of how image arrived
    canvas_meta = get_canvas_meta(image)
    tile_labels = canvas_meta.get("tile_labels") if canvas_meta else None

    for row in range(n_rows):
        for col in range(n_cols):
            y, x = row * tile, col * tile
            patch = img[y:y+tile, x:x+tile].astype(np.float32) / 255.0
            patch = np.expand_dims(patch, 0)
            probs      = model.predict(patch, verbose=0)[0]
            pred_idx   = np.argmax(probs)
            pred_class = CLASSES[pred_idx]
            confidence = float(probs[pred_idx])

            color_layer[y:y+tile, x:x+tile] = CLASS_COLORS[pred_class]
            class_counts[pred_class]          += 1
            class_confidences[pred_class].append(confidence)
            true_class = tile_labels[row * n_cols + col] if tile_labels and (row * n_cols + col) < len(tile_labels) else None
            predictions_grid[row][col] = (pred_class, confidence, true_class)
            total_tiles += 1
            last_probs   = probs
            last_class   = pred_class

    # Draw red borders on incorrect tiles only
    if tile_labels:
        color_layer_display = draw_correctness_borders(
            color_layer, predictions_grid, tile_labels, n_cols
        )
    else:
        color_layer_display = color_layer

    blended = blend(img, color_layer_display, opacity)

    # Ground truth accuracy summary
    if tile_labels:
        correct = sum(
            1 for row in range(n_rows) for col in range(n_cols)
            if (row * n_cols + col) < len(tile_labels)
            and predictions_grid[row][col][0] == tile_labels[row * n_cols + col]
        )
        truth_summary = (
            f"Correct: {correct}/{total_tiles} tiles "
            f"({correct/total_tiles*100:.1f}%)  |  🔴 = incorrect tile"
        )
    else:
        truth_summary = "No ground truth available for uploaded images"

    # Last tile summary
    last_confidence = float(last_probs[np.argmax(last_probs)]) * 100
    last_summary    = f"{last_class}  —  {last_confidence:.1f}% confidence"

    # Most common class summary
    most_common    = max(class_counts, key=class_counts.get)
    avg_conf       = np.mean(class_confidences[most_common]) * 100
    pct_tiles      = class_counts[most_common] / total_tiles * 100
    common_summary = (
        f"{most_common}  —  "
        f"{class_counts[most_common]} tiles ({pct_tiles:.0f}%)  |  "
        f"avg confidence {avg_conf:.1f}%"
    )

    # Bar chart DataFrame
    data_rows = [
        {"Class": cls, "Percentage (%)": round(count / total_tiles * 100, 1)}
        for cls, count in class_counts.items()
        if count > 0
    ]
    df = pd.DataFrame(data_rows).sort_values("Percentage (%)", ascending=False)

    cropped_original = image.crop((0, 0, w_crop, h_crop))

    return (
        cropped_original,
        blended,
        truth_summary,
        last_summary,
        common_summary,
        df,
        predictions_grid,
        img,
        color_layer_display,
    )
 
 
def update_opacity(opacity, original_arr, color_arr):
    """Re-blends stored arrays when the slider moves without re-classifying."""
    if original_arr is None or color_arr is None:
        return None
    return blend(original_arr, color_arr, opacity)
 
 
def on_tile_click(evt: gr.SelectData, predictions_grid):
    """Called when user clicks on the classification map. Shows that tile's result."""
    if predictions_grid is None:
        return ""
    x, y   = evt.index
    col    = x // 64
    row    = y // 64
    n_rows = len(predictions_grid)
    n_cols = len(predictions_grid[0]) if n_rows > 0 else 0
    if row >= n_rows or col >= n_cols:
        return ""
    pred_class, confidence, true_class = predictions_grid[row][col]
    result = f"Predicted: {pred_class}  —  {confidence*100:.1f}% confidence"
    if true_class is not None:
        if pred_class == true_class:
            result += f"  |  ✅ Correct  (true: {true_class})"
        else:
            result += f"  |  ❌ Wrong  (true: {true_class})"
    return result
 
 
def make_legend_html():
    items = ""
    for class_name, color in CLASS_COLORS.items():
        r, g, b = color
        items += f"""
            <div style="display:flex; align-items:center; margin:4px 0;">
                <div style="width:20px; height:20px; background:rgb({r},{g},{b});
                            margin-right:8px; border:1px solid #ccc; flex-shrink:0;">
                </div>
                <span>{class_name}</span>
            </div>
        """
    return f"""
        <div style="display:grid; grid-template-columns:1fr 1fr;
                    column-gap:24px; padding:8px;">
            {items}
        </div>
    """
 
 
with gr.Blocks(title="EuroSAT Land Cover Classifier") as app:
    gr.Markdown("# EuroSAT Land Cover Classifier")
    gr.Markdown(
        "Upload a satellite image (minimum 64×64 pixels) or choose one of the example canvases. "
        "It will be divided into 64×64 tiles and each tile classified "
        "into one of 10 land cover types. "
        "Click any tile on the classification map to see its prediction. "
        "For example canvases, green borders indicate correct predictions and red borders incorrect ones. "
        "The image will be cropped to the nearest multiple of 64 pixels."
    )

    predictions_state  = gr.State(None)
    original_arr_state = gr.State(None)
    color_arr_state    = gr.State(None)

    inp = gr.Image(type="pil", label="Input image")

    gr.Examples(
        examples=EXAMPLE_PATHS,
        inputs=inp,
        label="Example canvases — click to load",
        examples_per_page=15,
    )

    opacity = gr.Slider(
        minimum=0, maximum=100, value=50, step=1,
        label="Overlay opacity (%) - can be changed after classification without re-running the model",
    )

    btn = gr.Button("Classify", variant="primary")

    with gr.Row():
        original_out = gr.Image(type="pil", label="Original (cropped)")
        map_out      = gr.Image(type="pil", label="Classification map — click a tile")

    truth_out        = gr.Textbox(label="Ground truth accuracy")
    clicked_tile_out = gr.Textbox(label="Clicked tile")

    with gr.Row():
        last_tile_out = gr.Textbox(label="Last tile prediction")
        common_out    = gr.Textbox(label="Most common class")

    bar_out = gr.BarPlot(
        value=None,
        x="Class",
        y="Percentage (%)",
        title="Land cover distribution (%)",
        tooltip=["Class", "Percentage (%)"],
        y_lim=[0, 100],
        sort=None,
    )

    gr.Markdown("### Legend")
    gr.HTML(make_legend_html())

    btn.click(
        fn=classify_image,
        inputs=[inp, opacity],
        outputs=[
            original_out,
            map_out,
            truth_out,
            last_tile_out,
            common_out,
            bar_out,
            predictions_state,
            original_arr_state,
            color_arr_state,
        ],
    )

    opacity.change(
        fn=update_opacity,
        inputs=[opacity, original_arr_state, color_arr_state],
        outputs=map_out,
    )

    map_out.select(
        fn=on_tile_click,
        inputs=predictions_state,
        outputs=clicked_tile_out,
    )

app.launch()