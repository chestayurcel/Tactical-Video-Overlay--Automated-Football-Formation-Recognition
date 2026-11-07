from ultralytics import YOLO

if __name__ == '__main__':

    # 1. Muat model terbaik dari hasil training sebelumnya (bukan model yolov8m.pt kosongan)
    model = YOLO('runs/detect/Hasil_Training_Taktik/YOLOv8_Model_Baru-3/weights/best.pt')

    # 2. Mulai proses fine-tuning dengan dataset baru
    # Karena data.yaml ada di root folder yang sama dengan script ini, panggil langsung namanya
    results = model.train(
        data='data.yaml', 
        epochs=30, 
        imgsz=640, 
        device=0,
        project='Hasil_Finetune', # Membuat folder baru khusus untuk hasil finetune
        name='Model_Dengan_Wasit'             # Nama sub-folder agar tidak tercampur dengan yang lama
    )