import csv
from collections import defaultdict

LOG = 'log_formasi_per_frame.csv'
GT  = 'gt_template.csv'
OUT = 'hasil_evaluasi.csv'
WINDOW_TRANSISI = 150

def baca_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        contoh = f.readline()
        f.seek(0)
        delim = ';' if contoh.count(';') > contoh.count(',') else ','
        return list(csv.DictReader(f, delimiter=delim))

def parse_form(s):
    s = (s or '').strip()
    if s in ('', '---', 'Menghitung...', 'UNLABELABLE', 'unknown'):
        return None
    return tuple(int(x) for x in s.split('-'))

# ---- prediksi resolusi penuh ----
pred_mem, pred_nomem, status = {}, {}, {}
for r in baca_csv(LOG):
    k = (int(r['frame']), int(r['team_id']))
    pred_mem[k]   = parse_form(r['label_output'])
    pred_nomem[k] = parse_form(r['kandidat'])
    status[k]     = (r['status'] or '').strip()

# ---- ground truth ----
gt = {}
for r in baca_csv(GT):
    g = (r['gt_formation'] or '').strip()
    if g == '':
        continue
    k = (int(r['frame']), int(r['team_id']))
    gt[k] = None if g.upper() == 'UNLABELABLE' else parse_form(g)

V = sorted(k for k, v in gt.items() if v is not None)
if not V:
    raise SystemExit('GT belum terisi.')

frames_per_team = defaultdict(list)
for (fr, tm) in V:
    frames_per_team[tm].append(fr)
for tm in frames_per_team:
    frames_per_team[tm].sort()

def dekat(a, b):
    """Exact atau geser 1 pemain antar lini bersebelahan."""
    return a == b or sum(abs(x - y) for x, y in zip(a, b)) == 2

def evaluasi(pred, nama):
    benar = sum(1 for k in V if pred.get(k) == gt[k])
    near  = sum(1 for k in V if pred.get(k) is not None and dekat(pred[k], gt[k]))
    cover = sum(1 for k in V if pred.get(k) is not None)
    fa_cov = (benar / cover) if cover else 0.0

    # --- ULCR: perubahan label di dalam interval fase (GT konstan) ---
    total_frame_interval = perubahan = 0
    for tm, frs in frames_per_team.items():
        for f1, f2 in zip(frs, frs[1:]):
            if gt[(f1, tm)] != gt[(f2, tm)]:
                continue
            total_frame_interval += f2 - f1
            for t in range(f1 + 1, f2 + 1):
                if pred.get((t, tm)) != pred.get((t - 1, tm)):
                    perubahan += 1
    ulcr = (perubahan / total_frame_interval * 100) if total_frame_interval else 0.0

    # --- SLF: label tampil salah tapi merupakan GT masa lalu tim tsb ---
    masa_lalu = defaultdict(set)
    slf = 0
    for k in V:
        p = pred.get(k)
        if p is not None and p != gt[k] and p in masa_lalu[k[1]]:
            slf += 1
        masa_lalu[k[1]].add(gt[k])
    slf = slf / len(V) * 100

    # --- Transisi GT antar sampel + delay resolusi penuh ---
    trans = terdeteksi = 0
    delays = []
    for tm, frs in frames_per_team.items():
        for i, (f1, f2) in enumerate(zip(frs, frs[1:])):
            if gt[(f1, tm)] == gt[(f2, tm)]:
                continue
            trans += 1
            target = gt[(f2, tm)]
            batas = WINDOW_TRANSISI
            if i + 2 < len(frs):
                batas = min(batas, frs[i + 2] - f2)
            for d in range(0, batas + 1):
                if pred.get((f2 + d, tm)) == target:
                    terdeteksi += 1
                    delays.append(d)
                    break
    tdr = (terdeteksi / trans * 100) if trans else 0.0
    rata_delay = (sum(delays) / len(delays)) if delays else 0.0

    return {
        'konfigurasi': nama,
        'instance_valid': len(V),
        'FA_exact_%': round(benar / len(V) * 100, 1),
        'FA_near_%': round(near / len(V) * 100, 1),
        'Coverage_%': round(cover / len(V) * 100, 1),
        'FA_cov_%': round(fa_cov * 100, 1),
        'ULCR_per100fr': round(ulcr, 2),
        'SLF_%': round(slf, 1),
        'transisi_GT': trans,
        'TDR_%': round(tdr, 1),
        'delay_rata2_frame': round(rata_delay, 1),
    }

hasil = [evaluasi(pred_mem,   'Dengan Temporal Memory (usulan)'),
         evaluasi(pred_nomem, 'Tanpa Memory (instantaneous decoding)')]

dist = defaultdict(int)
for k in V:
    dist[status.get(k, '?')] += 1

with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['=== STATISTIK GROUND TRUTH ==='])
    w.writerow(['instance_valid_V', len(V)])
    w.writerow([])
    w.writerow(['=== HASIL METRIK ==='])
    w.writerow(hasil[0].keys())
    for h in hasil:
        w.writerow(h.values())
    w.writerow([])
    w.writerow(['=== DISTRIBUSI STATUS ==='])
    for s, n in sorted(dist.items()):
        w.writerow([s, n])

for h in hasil:
    print(f"--- {h['konfigurasi']} ---")
    for k, v in h.items():
        if k != 'konfigurasi':
            print(f"  {k:20s}: {v}")
    print()
print('Distribusi status:', dict(dist))
print(f'✅ Tersimpan: {OUT}')