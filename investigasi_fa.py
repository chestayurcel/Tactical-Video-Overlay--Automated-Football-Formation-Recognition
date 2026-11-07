import csv
from collections import defaultdict, Counter

LOG = 'log_formasi_per_frame.csv'
GT  = 'gt_template.csv'
OUT = 'investigasi_fa.csv'

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

def fmt(t):
    return '-'.join(map(str, t)) if t else 'NONE'

def balik(t):
    return (t[2], t[1], t[0])

# ---- muat prediksi resolusi penuh ----
pred, status, kand = {}, {}, {}
for r in baca_csv(LOG):
    k = (int(r['frame']), int(r['team_id']))
    pred[k]   = parse_form(r['label_output'])
    status[k] = (r['status'] or '').strip()
    kand[k]   = parse_form(r['kandidat'])

# ---- muat ground truth ----
gt, kat = {}, {}
for r in baca_csv(GT):
    g = (r['gt_formation'] or '').strip()
    if g == '':
        continue
    k = (int(r['frame']), int(r['team_id']))
    gt[k] = None if g.upper() == 'UNLABELABLE' else parse_form(g)
    kat[k] = (r.get('kategori') or '').strip().upper()

V = sorted(k for k, v in gt.items() if v is not None)

jenis_counter = Counter()
cross = Counter()          # (status, jenis)
conf = Counter()           # (gt, pred)
contoh_per_jenis = defaultdict(list)
rows_out = []
masa_lalu = defaultdict(set)
benar_per_tim = Counter()
total_per_tim = Counter()

for k in V:
    fr, tm = k
    g, p = gt[k], pred.get(k)
    total_per_tim[tm] += 1
    if p == g:
        jenis = 'BENAR'
        benar_per_tim[tm] += 1
    elif p is None:
        jenis = 'ABSTAIN'
    elif p == balik(g):
        jenis = 'FLIP_ORIENTASI'
    elif p in masa_lalu[tm]:
        jenis = 'STALE'
    elif sum(abs(a - b) for a, b in zip(p, g)) == 2:
        jenis = 'GESER_1_PEMAIN'
    else:
        jenis = 'STRUKTURAL'
    masa_lalu[tm].add(g)

    st = status.get(k, '?')
    jenis_counter[jenis] += 1
    cross[(st, jenis)] += 1
    if jenis != 'BENAR':
        conf[(g, p)] += 1
        if len(contoh_per_jenis[jenis]) < 8:
            contoh_per_jenis[jenis].append(fr)
        rows_out.append([fr, tm, kat[k], st, fmt(g), fmt(p), fmt(kand.get(k)), jenis])

# ================= LAPORAN =================
print('=' * 62)
print('DISTRIBUSI JENIS ERROR (|V| = %d)' % len(V))
print('=' * 62)
for j, n in jenis_counter.most_common():
    print(f'  {j:16s}: {n:5d}  ({n/len(V)*100:5.1f}%)')

print()
print('AKURASI PER TIM:')
for tm in sorted(total_per_tim):
    print(f'  tim{tm}: {benar_per_tim[tm]/total_per_tim[tm]*100:.1f}%')

print()
print('=' * 62)
print('CROSS-TAB: STATUS STATE MACHINE x JENIS ERROR')
print('=' * 62)
for (st, j), n in sorted(cross.items()):
    print(f'  {st:8s} | {j:16s} : {n}')

print()
print('=' * 62)
print('TOP 12 PASANG KEBINGUNGAN (GT -> PREDIKSI)')
print('=' * 62)
for (g, p), n in conf.most_common(12):
    print(f'  {fmt(g):8s} -> {fmt(p):8s} : {n}')

print()
print('CONTOH FRAME PER JENIS ERROR (untuk dicek visual):')
for j in ('FLIP_ORIENTASI', 'STALE', 'GESER_1_PEMAIN', 'STRUKTURAL', 'ABSTAIN'):
    if contoh_per_jenis[j]:
        print(f'  {j:16s}: frame {contoh_per_jenis[j]}')

with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['frame', 'team_id', 'kategori', 'status', 'gt', 'pred', 'kandidat', 'jenis_error'])
    w.writerows(rows_out)
print(f'\n✅ Rincian per instance tersimpan: {OUT}')