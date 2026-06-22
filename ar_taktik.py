import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

print("=== MEMULAI AR OVERLAY: DETEKSI FORMASI OTOMATIS ===")
model_path = r'C:\Users\HP\Semester 6\AI Terapan\Football_AI_Project\runs\detect\Hasil_Training_Taktik\YOLOv8_Model_Baru-3\weights\best.pt'
model = YOLO(model_path)

video_path = 'video_taktik_utama.mp4' 
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out_video = cv2.VideoWriter('ar_taktik_formasi_teks.avi', fourcc, fps, (width, height))

def get_dominant_color(image):
    image = cv2.resize(image, (20, 20))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0]

kmeans_teams = KMeans(n_clusters=2, random_state=42, n_init=10)
is_team_model_trained = False

print("\n=== MEMPROSES TEKS FORMASI 4-3-3 / 4-4-2... ===")

memori_formasi_kiri = "Formasi: Menghitung..."
memori_formasi_kanan = "Formasi: Menghitung..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, conf=0.15, iou=0.6, verbose=False)[0]
    
    player_colors = []
    player_data = [] 
    goalkeeper_data = []

    if results.boxes is not None:
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            foot_x, foot_y = int((x1 + x2) / 2), int(y2)
            
            if cls_name == 'player': 
                roi = frame[y1:y1 + h//2, x1 + w//4 : x2 - w//4]
                if roi.size > 0:
                    color = get_dominant_color(roi)
                    player_colors.append(color)
                    player_data.append({'foot': (foot_x, foot_y)})
            
            elif cls_name == 'goalkeeper':
                goalkeeper_data.append({'foot': (foot_x, foot_y)})
        
        if len(player_colors) >= 2 and not is_team_model_trained:
            kmeans_teams.fit(player_colors)
            is_team_model_trained = True
            
        if is_team_model_trained and len(player_colors) > 0:
            teams = kmeans_teams.predict(player_colors)
            for i, p in enumerate(player_data):
                p['team'] = teams[i]
                
            # Menentukan siapa tim Kiri dan siapa tim Kanan berdasarkan rata-rata sumbu X
            team_0_players = [p for p in player_data if p['team'] == 0]
            team_1_players = [p for p in player_data if p['team'] == 1]
            
            avg_x_0 = np.mean([p['foot'][0] for p in team_0_players]) if team_0_players else 0
            avg_x_1 = np.mean([p['foot'][0] for p in team_1_players]) if team_1_players else width
            team_0_is_left = avg_x_0 < avg_x_1

            for team_id, line_color in [(0, (0, 0, 200)), (1, (200, 0, 0))]:
                team_players = team_0_players if team_id == 0 else team_1_players
                is_left_team = (team_id == 0) if team_0_is_left else (team_id == 1)
                
                if len(team_players) >= 3:
                    xs = np.array([p['foot'][0] for p in team_players]).reshape(-1, 1)
                    kmeans_lines = KMeans(n_clusters=3, random_state=42, n_init=10)
                    line_labels = kmeans_lines.fit_predict(xs)
                    
                    # 1. PENGURUTAN LINI (BEK -> TENGAH -> DEPAN)
                    centers = kmeans_lines.cluster_centers_.flatten()
                    sorted_idx = np.argsort(centers)
                    if not is_left_team:
                        sorted_idx = sorted_idx[::-1] # Tim kanan, bek ada di X terbesar

                    linis = {0: [], 1: [], 2: []}
                    for idx, label in enumerate(line_labels):
                        linis[label].append(team_players[idx])
                    
                    # 2. MENGHITUNG TEKS FORMASI DENGAN VALIDASI 10 PEMAIN
                    formation_counts = []
                    total_pemain_terdeteksi = 0
                    
                    for cluster_idx in sorted_idx:
                        jumlah_di_lini = len(linis[cluster_idx])
                        formation_counts.append(str(jumlah_di_lini))
                        total_pemain_terdeteksi += jumlah_di_lini
                    
                    teks_formasi_sementara = f"Formasi: {'-'.join(formation_counts)}"
                    
                    # Menuliskan Teks di Layar (Kiri Atas untuk tim Kiri, Kanan Atas untuk tim Kanan)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    
                    if is_left_team:
                        # Hanya perbarui memori jika tepat 10 pemain terdeteksi
                        if total_pemain_terdeteksi == 10:
                            memori_formasi_kiri = teks_formasi_sementara
                        
                        cv2.putText(frame, memori_formasi_kiri, (50, 70), font, 1.2, (0, 0, 0), 4)
                        cv2.putText(frame, memori_formasi_kiri, (50, 70), font, 1.2, line_color, 2)
                    
                    else:
                        # Hanya perbarui memori jika tepat 10 pemain terdeteksi
                        if total_pemain_terdeteksi == 10:
                            memori_formasi_kanan = teks_formasi_sementara
                            
                        (text_w, _), _ = cv2.getTextSize(memori_formasi_kanan, font, 1.2, 4)
                        cv2.putText(frame, memori_formasi_kanan, (width - text_w - 50, 70), font, 1.2, (0, 0, 0), 4)
                        cv2.putText(frame, memori_formasi_kanan, (width - text_w - 50, 70), font, 1.2, line_color, 2)

                    # 3. MENGGAMBAR GARIS LINI
                    for label, kawan_lini in linis.items():
                        if len(kawan_lini) > 1:
                            kawan_lini.sort(key=lambda p: p['foot'][1])
                            for j in range(len(kawan_lini) - 1):
                                pt1 = kawan_lini[j]['foot']
                                pt2 = kawan_lini[j+1]['foot']
                                cv2.line(frame, pt1, pt2, line_color, 2)
            
            # 4. MENGGAMBAR LINGKARAN PEMAIN
            for p in player_data:
                ring_color = (0, 0, 255) if p['team'] == 0 else (255, 0, 0)
                fx, fy = p['foot']
                cv2.ellipse(frame, (fx, fy), (24, 10), 0, 0, 360, (0, 0, 0), 3)
                cv2.ellipse(frame, (fx, fy), (22, 8), 0, 0, 360, ring_color, 2)

        # 5. MENGGAMBAR KIPER
        for gk in goalkeeper_data:
            gfx, gfy = gk['foot']
            cv2.ellipse(frame, (gfx, gfy), (24, 10), 0, 0, 360, (0, 0, 0), 3)
            cv2.ellipse(frame, (gfx, gfy), (22, 8), 0, 0, 360, (0, 255, 255), 2)

    out_video.write(frame)

cap.release()
out_video.release()
print("\n✅ DETEKSI FORMASI SELESAI! Silakan cek 'ar_taktik_formasi_teks.avi'.")