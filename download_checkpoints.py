import gdown
import os

files = {
    "simclr_full_model": "17YvDD_m9C-M1ff5F1KV5QX4AbztE-PCJ",
    "simclr_encoder": "1Zi28MWvv2elUzldFZPHf-VunZDeCTSc6",
    "mae_full_model": "1bi7JpSUkEIJakcZYVFUDtqGYvUG2-ta8",
    "mae_encoder": "1vocGhf75i3o7XJAaw-LpYjH66pNScNIk",
    "mae_linear": "1YuWhf246pUSXlFFyfjLNL6bq5-Mq97FZ"
}

os.makedirs("checkpoints", exist_ok=True)

for name, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path = f"checkpoints/{name}.pth"
    gdown.download(url, out_path, quiet=False)
