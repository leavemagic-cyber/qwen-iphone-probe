"""T109b: real Core ML vision.mlpackage features for one shard of public COCO train2017 photos.
usage: run_shard.py <package> <shard_list.txt> <img_dir> <out_prefix>
Each photo: PIL open -> RGB -> bicubic 448x448 -> patchify -> predict (CPU_ONLY).
Writes <out_prefix>_feat.npy float16 [n,196,8192] = [merger|ds5|ds11|ds17] and <out_prefix>_meta.json (ids, pixel sha256, ms)."""
import hashlib, json, sys, time, platform, signal
import numpy as np
from PIL import Image
import coremltools as ct

pkg, lst, imgdir, prefix = sys.argv[1:5]
signal.alarm(2700)  # per-attempt cap (macOS has no timeout(1)); SIGALRM kills the process
ids = open(lst).read().split()


def patchify(a):
    x = (a.astype(np.float32) / 255 - .5) / .5
    x = np.broadcast_to(np.transpose(x, (2, 0, 1))[None], (2, 3, 448, 448))
    gh = gw = 28
    x = x.reshape(1, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(x).reshape(gh * gw, 1536).astype(np.float16)


m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_ONLY)
feat = np.zeros((len(ids), 196, 8192), np.float16)
sha, ms = [], []
t00 = time.time()
for k, i in enumerate(ids):
    a = np.asarray(Image.open('%s/%s.jpg' % (imgdir, i)).convert('RGB').resize((448, 448), Image.Resampling.BICUBIC), dtype=np.uint8)
    sha.append(hashlib.sha256(a.tobytes()).hexdigest())
    t0 = time.time()
    o = m.predict({'pixel_values': patchify(a)})
    ms.append(round((time.time() - t0) * 1000))
    feat[k] = np.concatenate([np.asarray(o[n], dtype=np.float32).reshape(196, 2048) for n in
                              ('hidden', 'deepstack_0', 'deepstack_1', 'deepstack_2')], axis=1).astype(np.float16)
    if k % 20 == 0:
        print(k, len(ids), ms[-1], round(time.time() - t00), flush=True)
assert np.isfinite(feat).all(), 'nonfinite'
np.save(prefix + '_feat.npy', feat)
json.dump(dict(ids=ids, pixel_sha256=sha, ms=ms, ct=ct.__version__, macos=platform.mac_ver()[0], wall_s=round(time.time() - t00)),
          open(prefix + '_meta.json', 'w'))
print('done', len(ids), 'wall', round(time.time() - t00))
