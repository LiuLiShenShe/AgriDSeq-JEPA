"""Read-only, full train/val audit; no test annotation access."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import numpy as np
import hashlib, csv, json, struct, time

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/data/datasets/PhenoBench/PhenoBench-v110'))
parser.add_argument('--reports', type=Path, default=Path('reports'))
args=parser.parse_args()
ROOT=args.root
REPORT=args.reports
REPORT.mkdir(exist_ok=True)

def audit_one(item):
 split,name=item
 ip=ROOT/split/'images'/name; mp=ROOT/split/'semantics'/name
 ib=ip.read_bytes();mb=mp.read_bytes()
 assert ib[:8]==mb[:8]==b'\x89PNG\r\n\x1a\n'
 ih=struct.unpack('>IIBB',ib[16:26]);mh=struct.unpack('>IIBB',mb[16:26])
 assert ih==(1024,1024,8,2),(ip,ih)
 assert mh==(1024,1024,16,0),(mp,mh)
 with Image.open(mp) as f:
  mask=np.asarray(f)
  assert mask.ndim==2 and np.issubdtype(mask.dtype,np.integer)
  assert mask.shape==(1024,1024)
  raw=np.bincount(mask.ravel(),minlength=5)
  assert len(raw)==5,(mp,len(raw))
  mapped=[int(raw[0]),int(raw[1]+raw[3]),int(raw[2]+raw[4])]
 return {
  'split':split,'filename':name,'image_bytes':len(ib),'image_sha256':hashlib.sha256(ib).hexdigest(),
  'mask_bytes':len(mb),'mask_sha256':hashlib.sha256(mb).hexdigest(),
  'width':1024,'height':1024,'mask_bit_depth':16,
  'soil_pixels':mapped[0],'crop_pixels':mapped[1],'weed_pixels':mapped[2],
  '_raw_pixels':raw.tolist()
 }

t0=time.monotonic();items=[]
for split,count in [('train',1407),('val',772)]:
 images=sorted(p.name for p in (ROOT/split/'images').glob('*.png'))
 masks=sorted(p.name for p in (ROOT/split/'semantics').glob('*.png'))
 assert len(images)==len(masks)==count
 assert images==masks
 items.extend((split,name) for name in images)
with ThreadPoolExecutor(max_workers=8) as pool:
 rows=list(pool.map(audit_one,items))
fields=[k for k in rows[0] if not k.startswith('_')]
manifest=REPORT/'phenobench_train_val_manifest.csv'
with manifest.open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for row in rows: writer.writerow({k:row[k] for k in fields})
summary={'root':str(ROOT),'version':'1.1.0','readme_sha256':hashlib.sha256((ROOT/'README.MD').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'test_images_count_only':len(list((ROOT/'test'/'images').glob('*.png'))),'splits':{},'audit_seconds':round(time.monotonic()-t0,3)}
for split in ('train','val'):
 subset=[r for r in rows if r['split']==split]
 mapped=np.sum([[r[k] for k in ('soil_pixels','crop_pixels','weed_pixels')] for r in subset],axis=0)
 raw=np.sum([r['_raw_pixels'] for r in subset],axis=0)
 summary['splits'][split]={'image_count':len(subset),'mask_count':len(subset),'raw_label_pixels':raw.tolist(),'mapped_label_pixels':mapped.tolist(),'mapped_percent':(100*mapped/mapped.sum()).tolist(),'all_image_headers':'1024x1024,8-bit RGB','all_mask_headers':'1024x1024,16-bit grayscale PNG'}
(REPORT/'phenobench_data_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
