import numpy as np
import nibabel as nib

def normalize(image):
    brain_mask = image > 0
    normalized = np.zeros_like(image, dtype=np.float32)
    if brain_mask.sum() > 0:
        mean = image[brain_mask].mean()
        std  = image[brain_mask].std()
        if std > 0:
            normalized[brain_mask] = (image[brain_mask] - mean) / std
    return normalized

def preprocess_patient(patient_path, patient_name):
    modalities = []
    for suffix in ['_t1.nii', '_t1ce.nii', '_t2.nii', '_flair.nii']:
        path = f"{patient_path}/{patient_name}{suffix}"
        img  = nib.load(path).get_fdata(dtype=np.float32)
        modalities.append(normalize(img))

    image = np.stack(modalities, axis=-1)
    mask  = nib.load(f"{patient_path}/{patient_name}_seg.nii").get_fdata().astype('int8')
    mask[mask == 4] = 3

    # Crop
    image = image[56:184, 56:184, 13:141, :]
    mask  = mask [56:184, 56:184, 13:141]

    return image, mask