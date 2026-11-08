import sys, glob, os, collections, cv2, numpy as np
sys.path.insert(0, sys.argv[1])
import stair_vision as sv
out = sys.argv[2]
dirs = ['backup/legacy_datasets/raw_photos/계단', 'backup/legacy_datasets/raw_photos/drive-download-20261105T045852Z-1-001']
files = [f for d in dirs for f in sorted(glob.glob(d + '/*.jpg'))]
sv.load_model()
rows = []
cls_count = {'dark': collections.Counter(), 'light': collections.Counter()}
for f in files:
    im = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
    s = 1280 / max(im.shape[:2]); im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bright = float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean())
    grp = 'dark' if bright < 70 else 'light'
    r = sv.analyze(im)
    for d in r['detections']: cls_count[grp][d['class']] += 1
    rows.append((os.path.basename(f), grp, round(bright,1), len(r['detections']), r['infer_ms']))
    name = os.path.splitext(os.path.basename(f))[0]
    if name in ('han01_0','han01_1','han02_0','han02_1'):
        cv2.imwrite(os.path.join(out, name + '_result.jpg'), sv.draw(im, r))
for g in ('dark','light'):
    rs = [x for x in rows if x[1]==g]
    print(f"{g}: {len(rs)}장, 평균밝기 {np.mean([x[2] for x in rs]):.1f}, 탐지된 사진 {sum(1 for x in rs if x[3]>0)}장, 총 탐지 {sum(x[3] for x in rs)}개, 클래스 {dict(cls_count[g])}")
ms = [x[4] for x in rows[1:]]
print(f"전체 {len(rows)}장, 평균 추론 {np.mean(ms):.1f} ms (중앙값 {np.median(ms):.1f})")
