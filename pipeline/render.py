"""Render a skinned textured sequence (skin_scan.write_buffers output) to an MP4 via three.js +
headless Chromium. Buffers are served over a localhost http server; three.js fetches them, we
step frame by frame and screenshot.

Chromium is launched with SwiftShader (CPU WebGL) and --no-sandbox/--disable-dev-shm-usage so it
runs on an HPC compute node with no GPU and no /dev/shm. three.min.js is vendored next to this
file - nothing is fetched from a CDN.

CLI:  python -m pipeline.render --buffers <skin_out dir> --out clip.mp4 [--loops 3] [--panel 1280]
"""
import os, io, json, base64, http.server, socketserver, threading, functools, time, asyncio, argparse
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
THREE_JS = os.path.join(HERE, 'three.min.js')

# fixed tripod camera - same framing as the untextured v5 pipeline: (1.8,-1.8,1.2) eye
# direction, Z up, distance solved for ~52% vertical fill of a ~1.8 m body.
FOV = 45
FILL = 0.52
OBJ_H = 1.8
_dist = OBJ_H / (2 * FILL * np.tan(np.radians(FOV / 2))) * 0.92     # 0.92: 3/4-view foreshortening
_eye = np.array([1.8, -1.8, 1.2]); _eye = _eye / np.linalg.norm(_eye) * _dist
_light = np.array([180, -60, 90]); _light = _light / np.linalg.norm(_light) * 10


def _html(meta, panel):
    three_src = open(THREE_JS).read()
    tex_b64 = base64.b64encode(open(os.path.join(BUFFERS, 'texture.jpg'), 'rb').read()).decode()
    nv, nf, n = meta['n_verts'], meta['n_faces'], meta['n_frames']
    return f"""<!doctype html><html><head><meta charset=utf-8>
<style>html,body{{margin:0;background:#000;overflow:hidden}}canvas{{display:block}}</style></head>
<body><script>{three_src}</script><script>
const NV={nv}, N={n}, PANEL={panel};
async function buf(u){{const r=await fetch(u);return await r.arrayBuffer();}}
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x000000);
const camera=new THREE.PerspectiveCamera({FOV},1,0.01,100);
camera.position.set({_eye[0]},{_eye[1]},{_eye[2]}); camera.up.set(0,0,1); camera.lookAt(0,0,0);
const renderer=new THREE.WebGLRenderer({{antialias:true,preserveDrawingBuffer:true}});
renderer.setSize(PANEL,PANEL); renderer.outputColorSpace=THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,0.8));
const k=new THREE.DirectionalLight(0xffffff,1.5); k.position.set({_light[0]},{_light[1]},{_light[2]}); scene.add(k);
const fl=new THREE.DirectionalLight(0xffffff,0.35); fl.position.set({-_light[0]},{-_light[1]},{_light[2]}); scene.add(fl);
let geom, positions;
async function init(){{
  const faces=new Uint32Array(await buf('faces.u32'));
  const uv=new Float32Array(await buf('uv.f32'));
  positions=new Float32Array(await buf('positions.f32'));
  const tex=await new Promise(res=>{{const im=new Image();
    im.onload=()=>{{const t=new THREE.Texture(im); t.colorSpace=THREE.SRGBColorSpace; t.needsUpdate=true; res(t);}};
    im.src="data:image/jpeg;base64,{tex_b64}";}});
  geom=new THREE.BufferGeometry();
  geom.setIndex(new THREE.BufferAttribute(faces,1));
  geom.setAttribute('uv',new THREE.BufferAttribute(uv,2));
  geom.setAttribute('position',new THREE.BufferAttribute(positions.slice(0,NV*3),3));
  geom.computeVertexNormals();
  sanitizeNormals(geom);
  scene.add(new THREE.Mesh(geom,new THREE.MeshStandardMaterial({{map:tex,roughness:0.85,metalness:0.0}})));
  window.__ready=true;
}}
// A zero-area triangle (degenerate mesh_decimation collapse, occasionally a wildly-quantized
// pose driving two vertices to the same spot) makes computeVertexNormals() divide by a
// near-zero length, producing NaN/Infinity normals - which SwiftShader (software WebGL) doesn't
// contain to the offending triangle, it corrupts the ENTIRE frame to solid white. Rare and hard
// to fully predict in advance (found via a param_quantization+mesh_decimation combination whose
// specific pose moved different triangles degenerate than the reference frame decimation was
// filtered against), so guard the renderer itself rather than chase every producing case:
// zero out any non-finite normal component after every computeVertexNormals() call. A zeroed
// normal just makes that one vertex unlit, not a corrupted frame.
function sanitizeNormals(g){{
  const n=g.getAttribute('normal').array;
  for(let i=0;i<n.length;i++) if(!isFinite(n[i])) n[i]=0;
}}
window.setFrame=function(i){{
  const o=i*NV*3;
  geom.getAttribute('position').array.set(positions.subarray(o,o+NV*3));
  geom.getAttribute('position').needsUpdate=true;
  geom.computeVertexNormals();
  sanitizeNormals(geom);
  renderer.render(scene,camera);
}};
init();
</script></body></html>"""


def _is_blank(img, thresh=0.98):
    """A near-solid-color frame - SwiftShader's texture upload is asynchronous and can lag the
    JS-side 'ready' signal by several real seconds regardless of frame count (found empirically:
    a clean PREFIX of blank frames, self-correcting partway through, unrelated to which
    distortion was applied - a warmup race, not a geometry bug). Used both to warm up the
    renderer before capturing anything real, and as a per-frame safety net after."""
    a = img.reshape(-1, img.shape[-1])
    return (np.abs(a.astype(np.int16) - a[0]).max(axis=1) < 4).mean() > thresh


async def _screenshot(pg):
    return np.array(Image.open(io.BytesIO(await pg.screenshot())).convert('RGB'))


async def _capture(port, n, panel):
    frames = []
    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True, args=[
            '--use-gl=swiftshader', '--enable-webgl', '--no-sandbox', '--disable-dev-shm-usage'])
        pg = await b.new_page(viewport={'width': panel, 'height': panel})
        pg.on('console', lambda m: print('  [browser]', m.text) if m.type == 'error' else None)
        await pg.goto(f'http://127.0.0.1:{port}/view.html')
        await pg.wait_for_function('window.__ready===true', timeout=120000)

        # warm up: keep re-rendering frame 0 until it's not blank, instead of a fixed guess at
        # how long SwiftShader's texture upload takes (varies with machine load / texture size)
        for _ in range(60):
            await pg.evaluate('window.setFrame(0)')
            shot = await _screenshot(pg)
            if not _is_blank(shot):
                break
            await pg.wait_for_timeout(200)

        for i in range(n):
            if i > 0:
                await pg.evaluate(f'window.setFrame({i})')
                shot = await _screenshot(pg)
            # per-frame safety net - retry a still-blank frame rather than bake it into the video
            retries = 0
            while _is_blank(shot) and retries < 20:
                await pg.wait_for_timeout(200)
                await pg.evaluate(f'window.setFrame({i})')
                shot = await _screenshot(pg)
                retries += 1
            frames.append(shot)
        await b.close()
    return frames


def capture_frames(buffers_dir, panel=1280):
    """Render every frame in buffers_dir (positions.f32/faces.u32/uv.f32/texture.jpg/meta.json)
    to RGB arrays. Split out from encode_video() so frame-space distortions (motion_blur) can
    post-process the raw frames before they're written to video."""
    global BUFFERS
    BUFFERS = buffers_dir
    meta = json.load(open(os.path.join(buffers_dir, 'meta.json')))
    open(os.path.join(buffers_dir, 'view.html'), 'w').write(_html(meta, panel))

    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
    httpd = socketserver.TCPServer(('127.0.0.1', 0), functools.partial(Q, directory=buffers_dir))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    t = time.time()
    frames = asyncio.run(_capture(port, meta['n_frames'], panel))
    httpd.shutdown()
    print(f'captured {len(frames)} frames in {time.time()-t:.0f}s', flush=True)
    return frames, meta


def encode_video(frames, out_path, fps, loops=3):
    seq = frames * loops
    h, w = seq[0].shape[:2]; w, h = w - w % 2, h - h % 2
    with imageio.get_writer(out_path, format='FFMPEG', fps=fps, codec='libx264',
                            output_params=['-crf', '18', '-pix_fmt', 'yuv420p']) as wr:
        for f in seq:
            wr.append_data(f[:h, :w])
    return len(seq)


def render(buffers_dir, out_path, loops=3, panel=1280):
    frames, meta = capture_frames(buffers_dir, panel)
    return encode_video(frames, out_path, meta['playback_fps'], loops)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--buffers', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--loops', type=int, default=3)
    ap.add_argument('--panel', type=int, default=1280)
    a = ap.parse_args()
    nf = render(a.buffers, a.out, a.loops, a.panel)
    print(f'wrote {a.out}  ({nf} frames)')
