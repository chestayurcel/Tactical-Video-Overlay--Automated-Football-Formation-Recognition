import cv2
import csv
import os
import numpy as np

VIDEO      = 'video_utama.mp4'
OUT_DIR    = 'gt_frames'
N_SAMPLE   = 1000
TEMPLATE   = 'gt_template.csv'

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)

# Sampling sistematis, konvensi 1-based (SAMA dengan log ar_taktik.py)
idxs = np.linspace(1, total, N_SAMPLE).round().astype(int)
idxs = sorted(set(idxs.tolist()))

os.makedirs(OUT_DIR, exist_ok=True)
with open(TEMPLATE, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame', 'detik', 'team_id', 'gt_formation', 'kategori'])
    for t in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t - 1)   # konversi ke 0-based
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(os.path.join(OUT_DIR, f'frame_{t:05d}.png'), frame)
        for team in (0, 1):
            w.writerow([t, round((t - 1) / fps, 2), team, '', ''])

cap.release()
print(f"✅ {len(idxs)} frame diekstrak ke folder '{OUT_DIR}/'")
print(f"✅ Template anotasi: '{TEMPLATE}' ({len(idxs)*2} baris)")
print("   Isi kolom gt_formation (mis. 4-3-3) dan kategori (L1/L2/L3).")
print("   Jika frame ambigu/transisi: isi gt_formation = UNLABELABLE, kategori = L3.")