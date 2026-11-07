import cv2
import numpy as np
import csv
import time
from collections import deque, Counter
from ultralytics import YOLO
from sklearn.cluster import KMeans

# =========================================================
# KONFIGURASI
# =========================================================
T_MAX = 150                      # batas latch (frame). 150 frame @30fps = 5 detik
REFRESH_INTERVAL = 30            # adaptasi centroid halus tiap 30 frame (1 dtk)
COLLAPSE_MAX = 12                # >12 pemain dalam satu klaster = mustahil
COLLAPSE_MIN = 4                 # <4 pemain saat total>=12 = indikasi kolaps
TRACK_WINDOW = 15                # v4: jendela voting identitas tim per track
LINE_WINDOW = 15                 # v5: jendela voting LINI per track (~0,5 dtk)
LOG_CSV = 'log_formasi_per_frame.csv'

VALID_FORMATIONS = {
    (3, 3, 4), (3, 4, 3), (3, 5, 2), (3, 6, 1),
    (4, 2, 4), (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 2, 3), (5, 3, 2), (5, 4, 1),
}

class TemporalMemory:
    """State machine memori temporal (Algoritma 1 pada paper)."""
    def __init__(self, t_max=T_MAX):
        self.label = None
        self.pernah_valid = False
        self.tau = 0
        self.t_max = t_max
        self.status = 'init'

    def update(self, kandidat, total):
        valid = (total == 10) and (kandidat is not None) and (kandidat in VALID_FORMATIONS)
        if valid:
            berubah = (kandidat != self.label)
            self.label = kandidat
            self.pernah_valid = True
            self.tau = 0
            self.status = 'update' if berubah else 'hold'
        else:
            self.tau += 1
            if self.pernah_valid and self.tau >= self.t_max:
                self.label = None
                self.status = 'unknown'
            elif not self.pernah_valid:
                self.status = 'init'
            else:
                self.status = 'latch'
        return self.status

    def text(self):
        if self.label is not None:
            return "Formasi: " + "-".join(str(n) for n in self.label)
        if self.pernah_valid:
            return "Formasi: ---"
        return "Formasi: Menghitung..."

# =========================================================
# SEGMENTASI WARNA INVARIANT-CAHAYA
# =========================================================
def ekstraksi_warna_invarian(roi):
    roi = cv2.resize(roi, (20, 20))
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)
    norma = rgb.sum(axis=2, keepdims=True) + 1e-6
    kroma = rgb / norma
    return kroma.reshape(-1, 3).mean(axis=0)

def perlu_retrain(labels):
    c0 = int(np.sum(labels == 0))
    c1 = int(np.sum(labels == 1))
    total = c0 + c1
    if c0 > COLLAPSE_MAX or c1 > COLLAPSE_MAX:
        return True
    if total >= 12 and (c0 < COLLAPSE_MIN or c1 < COLLAPSE_MIN):
        return True
    return False

def selaraskan_identitas(centroids_baru, centroids_lama):
    d_tetap = (np.linalg.norm(centroids_baru[0] - centroids_lama[0]) +
               np.linalg.norm(centroids_baru[1] - centroids_lama[1]))
    d_tukar = (np.linalg.norm(centroids_baru[0] - centroids_lama[1]) +
               np.linalg.norm(centroids_baru[1] - centroids_lama[0]))
    return d_tukar < d_tetap

# =========================================================
# INISIALISASI
# =========================================================
print("=== MEMULAI AR OVERLAY: DETEKSI FORMASI OTOMATIS (v5) ===")
model = YOLO('runs/detect/runs/detect/Hasil_Finetune/Model_Dengan_Wasit-3/weights/best.pt')

video_path = 'video_utama.mp4'
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('tactical_formation.avi', fourcc, fps, (width, height))

kmeans_teams = KMeans(n_clusters=2, random_state=42, n_init=10)
is_team_model_trained = False
track_history = {}               # v4: track_id -> deque label tim mentah
line_history = {}                # v5: (team_id, track_id) -> deque lini ordinal
last_voted_line = {}             # v5: memori tie-break (histeresis)

memori = {0: TemporalMemory(), 1: TemporalMemory()}

log_file = open(LOG_CSV, 'w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['frame', 'team_id', 'n_outfield', 'kandidat', 'status', 'label_output'])

frame_idx = 0
print("\n=== MEMPROSES: MEMORI TEMPORAL + WARNA ADAPTIF + VOTING TIM + VOTING LINI ===")

t0 = time.perf_counter()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          imgsz=640, conf=0.15, iou=0.6, verbose=False)[0]

    player_colors = []
    player_data = []
    goalkeeper_data = []

    if results.boxes is not None:
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name == 'referee':
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            foot_x, foot_y = int((x1 + x2) / 2), int(y2)
            tid = int(box.id[0]) if box.id is not None else None

            if cls_name == 'player':
                yy1 = y1 + int(h * 0.25)
                yy2 = y1 + int(h * 0.65)
                if yy2 - yy1 < 2:
                    yy1, yy2 = y1, y1 + max(2, h // 2)
                roi = frame[yy1:yy2, x1 + w//4 : x2 - w//4]
                if roi.size > 0:
                    color = ekstraksi_warna_invarian(roi)
                    player_colors.append(color)
                    player_data.append({'foot': (foot_x, foot_y), 'tid': tid})
            elif cls_name == 'goalkeeper':
                goalkeeper_data.append({'foot': (foot_x, foot_y)})

        # --- PELATIHAN / ADAPTASI CENTROID TIM ---
        if len(player_colors) >= 2 and not is_team_model_trained:
            kmeans_teams.fit(player_colors)
            is_team_model_trained = True
            track_history.clear()

        if is_team_model_trained and len(player_colors) > 0:
            if frame_idx % REFRESH_INTERVAL == 0 and len(player_colors) >= 2:
                km_baru = KMeans(n_clusters=2, init=kmeans_teams.cluster_centers_.copy(),
                                 n_init=1, random_state=42)
                raw = km_baru.fit_predict(player_colors)
                kmeans_teams = km_baru
            else:
                raw = kmeans_teams.predict(player_colors)

            # --- v4: VOTING MAYORITAS IDENTITAS TIM PER TRACK ---
            voted = []
            for i, p in enumerate(player_data):
                tid = p['tid']
                r = int(raw[i])
                if tid is None:
                    voted.append(r)
                    continue
                hist = track_history.setdefault(tid, deque(maxlen=TRACK_WINDOW))
                hist.append(r)
                voted.append(Counter(hist).most_common(1)[0][0])
            voted = np.array(voted)

            # --- deteksi kolaps + retrain + penjajaran identitas ---
            if len(player_colors) >= 2 and perlu_retrain(voted):
                centroids_lama = kmeans_teams.cluster_centers_.copy()
                kmeans_teams.fit(player_colors)
                raw = kmeans_teams.predict(player_colors)
                if selaraskan_identitas(kmeans_teams.cluster_centers_, centroids_lama):
                    raw = 1 - raw
                track_history.clear()
                line_history.clear()          # v5: reset smoothing lini juga
                last_voted_line.clear()
                for i, p in enumerate(player_data):
                    if p['tid'] is not None:
                        track_history[p['tid']] = deque([int(raw[i])], maxlen=TRACK_WINDOW)
                voted = np.array([int(x) for x in raw])

            teams = voted
            for i, p in enumerate(player_data):
                p['team'] = int(teams[i])

            team_0_players = [p for p in player_data if p['team'] == 0]
            team_1_players = [p for p in player_data if p['team'] == 1]

            avg_x_0 = np.mean([p['foot'][0] for p in team_0_players]) if team_0_players else 0
            avg_x_1 = np.mean([p['foot'][0] for p in team_1_players]) if team_1_players else width
            team_0_is_left = avg_x_0 < avg_x_1

            for team_id, line_color in [(0, (0, 0, 200)), (1, (200, 0, 0))]:
                team_players = team_0_players if team_id == 0 else team_1_players
                is_left_team = (team_id == 0) if team_0_is_left else (team_id == 1)

                kandidat = None
                linis = None
                total = len(team_players)

                if total >= 3:
                    xs = np.array([p['foot'][0] for p in team_players]).reshape(-1, 1)
                    if len(np.unique(xs)) >= 3:      # v5: guard titik duplikat
                        kmeans_lines = KMeans(n_clusters=3, random_state=42, n_init=10)
                        line_labels = kmeans_lines.fit_predict(xs)

                        centers = kmeans_lines.cluster_centers_.flatten()
                        sorted_idx = np.argsort(centers)
                        if not is_left_team:
                            sorted_idx = sorted_idx[::-1]
                        order = [int(c) for c in sorted_idx]   # order[ordinal] = id klaster

                        # --- v5: VOTING LINI PER TRACK (ordinal 0=bek,1=tengah,2=depan) ---
                        linis = {0: [], 1: [], 2: []}
                        for idx_p, lab in enumerate(line_labels):
                            ordinal = order.index(int(lab))
                            p = team_players[idx_p]
                            tid = p['tid']
                            if tid is None:
                                voted_line = ordinal
                            else:
                                key = (team_id, tid)
                                hist = line_history.setdefault(key, deque(maxlen=LINE_WINDOW))
                                hist.append(ordinal)
                                top = Counter(hist).most_common()
                                best = top[0][1]
                                cands = [l for l, c in top if c == best]
                                prev = last_voted_line.get(key)
                                voted_line = prev if prev in cands else min(cands)
                                last_voted_line[key] = voted_line
                            p['line'] = voted_line
                            linis[voted_line].append(p)
                        kandidat = (len(linis[0]), len(linis[1]), len(linis[2]))

                status = memori[team_id].update(kandidat, total)
                teks = memori[team_id].text()

                log_writer.writerow([
                    frame_idx, team_id, total,
                    "-".join(map(str, kandidat)) if kandidat else "",
                    status,
                    teks.replace("Formasi: ", "")
                ])

                font = cv2.FONT_HERSHEY_DUPLEX
                if is_left_team:
                    cv2.putText(frame, teks, (50, 70), font, 1.2, (0, 0, 0), 4)
                    cv2.putText(frame, teks, (50, 70), font, 1.2, line_color, 2)
                else:
                    (tw, _), _ = cv2.getTextSize(teks, font, 1.2, 4)
                    cv2.putText(frame, teks, (width - tw - 50, 70), font, 1.2, (0, 0, 0), 4)
                    cv2.putText(frame, teks, (width - tw - 50, 70), font, 1.2, line_color, 2)

                if linis is not None:
                    for lab, kawan_lini in linis.items():
                        if len(kawan_lini) > 1:
                            kawan_lini.sort(key=lambda p: p['foot'][1])
                            for j in range(len(kawan_lini) - 1):
                                cv2.line(frame, kawan_lini[j]['foot'],
                                         kawan_lini[j+1]['foot'], line_color, 2)

            for p in player_data:
                ring_color = (0, 0, 255) if p['team'] == 0 else (255, 0, 0)
                fx, fy = p['foot']
                cv2.ellipse(frame, (fx, fy), (24, 10), 0, 0, 360, (0, 0, 0), 3)
                cv2.ellipse(frame, (fx, fy), (22, 8), 0, 0, 360, ring_color, 2)

        for gk in goalkeeper_data:
            gfx, gfy = gk['foot']
            cv2.ellipse(frame, (gfx, gfy), (24, 10), 0, 0, 360, (0, 0, 0), 3)
            cv2.ellipse(frame, (gfx, gfy), (22, 8), 0, 0, 360, (0, 255, 255), 2)

    out_video.write(frame)

cap.release()
out_video.release()
log_file.close()
print(f"\n✅ SELESAI! {frame_idx} frame diproses.")
print(f"📄 Log evaluasi: {LOG_CSV}")
print("🎬 Video output: tactical_formation.avi")
print(f"⏱️ FPS full-pipeline: {frame_idx / (time.perf_counter() - t0):.1f}")