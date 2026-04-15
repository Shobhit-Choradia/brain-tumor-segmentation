from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64, io, os, nibabel as nib
from model_utils import normalize
import shutil

# Prevent GPU issues on Render
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)
CORS(app)

# ── Custom functions ─────────────────────────────────────────────
def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2])
    return tf.reduce_mean((2.*intersection+smooth)/(union+smooth))

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    return dice_loss(y_true, y_pred) + ce

def dice_class(index):
    def dice(y_true, y_pred):
        return dice_coef(y_true[:,:,:,index:index+1], y_pred[:,:,:,index:index+1])
    dice.__name__ = f"dice_class_{index}"
    return dice

# ── Model Lazy Loading ───────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model (3).keras")
model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = load_model(
            MODEL_PATH,
            custom_objects={
                "combined_loss": combined_loss,
                "dice_coef": dice_coef,
                "dice_class_0": dice_class(0),
                "dice_class_1": dice_class(1),
                "dice_class_2": dice_class(2),
                "dice_class_3": dice_class(3),
            }
        )
    return model

# ── Helper ───────────────────────────────────────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return result

# ── Routes ───────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        files = request.files.getlist('file')

        if not files or len(files) < 4:
            return jsonify({"error": "Upload 4 MRI files"}), 400

        tmp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(tmp_dir, exist_ok=True)

        patient = None

        # Save uploaded files
        for file in files:
            filename = file.filename
            path = os.path.join(tmp_dir, filename)
            file.save(path)

            if patient is None:
                parts = filename.replace('.nii','').split('_')
                patient = f"{parts[0]}_{parts[1]}_{parts[2]}"

        # Load modalities
        modalities = []
        for suffix in ['_t1.nii','_t1ce.nii','_t2.nii','_flair.nii']:
            path = os.path.join(tmp_dir, f"{patient}{suffix}")

            if not os.path.exists(path):
                return jsonify({"error": f"Missing file: {patient}{suffix}"}), 400

            img = nib.load(path).get_fdata(dtype=np.float32)
            modalities.append(normalize(img))

        # Prepare input
        image = np.stack(modalities, axis=-1)
        image = image[56:184, 56:184, 13:141, :]

        slice_idx = image.shape[2] // 2
        image_slice = image[:,:,slice_idx,:][np.newaxis,...]

        # Load model properly
        model_instance = get_model()

        # Prediction
        pred = model_instance.predict(image_slice, verbose=0)
        pred_mask = np.argmax(pred[0], axis=-1)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(10,5), facecolor='#1a1a2e')

        axes[0].imshow(image_slice[0,:,:,1].T, cmap='gray', origin='lower')
        axes[0].set_title("MRI", color='white')
        axes[0].axis('off')

        axes[1].imshow(pred_mask.T, cmap='jet', origin='lower')
        axes[1].set_title("Prediction", color='white')
        axes[1].axis('off')

        seg_image = fig_to_base64(fig)

        # Volume estimation (optimized)
        label_counts = {0:0,1:0,2:0,3:0}

        for s in range(0, image.shape[2], max(1, image.shape[2]//20)):
            p = model_instance.predict(image[:,:,s,:][np.newaxis,...], verbose=0)
            pm = np.argmax(p[0], axis=-1)

            for l in range(4):
                label_counts[l] += int((pm==l).sum())

        vols = {k: round(v/1000,2) for k,v in label_counts.items()}
        total = round(vols[1] + vols[2] + vols[3], 2)

        severity = "HIGH" if total>50 else "MODERATE" if total>20 else "LOW"

        # Cleanup temp folder
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return jsonify({
            "segmentation_image": seg_image,
            "volumes": {
                "necrotic": vols[1],
                "edema": vols[2],
                "enhancing": vols[3],
                "total": total
            },
            "severity": severity,
            "slice_idx": int(slice_idx)
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

# ── Run ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)