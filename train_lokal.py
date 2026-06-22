from roboflow import Roboflow
from ultralytics import YOLO

if __name__ == '__main__':

    print("=== TAHAP 1: MENGUNDUH DATASET ===")
    # Masukkan API Key Roboflow milikmu
    rf = Roboflow(api_key="juJ2GqPRazD3j6KoNysF")
    project = rf.workspace("alfonso-secretara-tcnica").project("football-tactical")
    version = project.version(4)
    dataset = version.download("yolov8")
                    

    print("\n=== TAHAP 2: MEMULAI PROSES TRAINING ===")
    # Menggunakan pre-trained YOLOv8 Medium
    model = YOLO('yolov8m.pt')

    # Proses training menggunakan RTX 4060
    results = model.train(
        data=f'{dataset.location}/data.yaml',
        epochs=70,            # Target 70 siklus agar AI paham objek kecil
        imgsz=640,            # Resolusi standar (ideal untuk dataset ini)
        batch=16,             # VRAM 8GB milikmu sangat kuat untuk memproses 16 gambar sekaligus
        device=0,             # Memaksa PyTorch menggunakan GPU (NVIDIA CUDA:0)
        project='Hasil_Training_Taktik', # Nama folder utama
        name='YOLOv8_Model_Baru'         # Nama sub-folder
    )

    print("\n✅ TRAINING SELESAI! Model best.pt telah tersimpan.")