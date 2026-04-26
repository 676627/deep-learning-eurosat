import gradio as gr
import numpy as np
import keras
from PIL import Image

model = keras.models.load_model("checkpoints/RGB_batchnorm_3conv_full_best.keras")

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

# One color per class for the output map
CLASS_COLORS = {
    "AnnualCrop":           (255, 200, 0),
    "Forest":               (0, 100, 0),
    "HerbaceousVegetation": (0, 200, 0),
    "Highway":              (128, 128, 128),
    "Industrial":           (200, 0, 200),
    "Pasture":              (180, 255, 100),
    "PermanentCrop":        (255, 100, 0),
    "Residential":          (255, 0, 0),
    "River":                (0, 0, 255),
    "SeaLake":              (0, 200, 255),
}

def classify_image(image):
    img = np.array(image)
    h, w = img.shape[:2]
    tile = 64

    # Crop to the largest multiple of 64 in each dimension
    h_crop = (h // tile) * tile
    w_crop = (w // tile) * tile
    img = img[:h_crop, :w_crop]

    # Build output colour map — same size as cropped input
    output = np.zeros((h_crop, w_crop, 3), dtype=np.uint8)

    # Classify each 64x64 tile
    for y in range(0, h_crop, tile):
        for x in range(0, w_crop, tile):
            patch = img[y:y+tile, x:x+tile]                # extract tile
            patch = patch.astype(np.float32) / 255.0       # normalise
            patch = np.expand_dims(patch, 0)               # add batch dim

            probs      = model.predict(patch, verbose=0)[0]
            pred_class = CLASSES[np.argmax(probs)]
            color      = CLASS_COLORS[pred_class]

            output[y:y+tile, x:x+tile] = color             # fill with class color

    # Build a legend
    legend_items = [(name, color) for name, color in CLASS_COLORS.items()]

    return Image.fromarray(output)

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
    return f'''
        <div style="display:grid; grid-template-columns:1fr 1fr; 
                    column-gap:24px; padding:8px;">
            {items}
        </div>
    '''

with gr.Blocks() as app:
    gr.Markdown("## EuroSAT Land Cover Classifier")
    gr.Markdown(
        "Upload a satellite image. It will be divided into 64×64 tiles "
        "and each tile classified into one of 10 land cover types."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Input satellite image")
        out = gr.Image(type="pil", label="Classification map")

    gr.Button("Classify").click(fn=classify_image, inputs=inp, outputs=out)

    gr.HTML(make_legend_html(), elem_id="legend")

app.launch()