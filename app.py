import os
import re
import uuid
import json
import time
import glob
import shutil
import threading
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, abort

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 60 MB cap (64 for headroom)

WORK_DIR = Path('/tmp/reverse_edit_jobs')
WORK_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.webm', '.m4v'}

# ── Job store ────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

def _job_set(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(kwargs)

def _job_get(job_id: str) -> dict:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))

# ── Easing functions ─────────────────────────────────────────────────────────
def ease(t: float, mode: str) -> float:
    t = max(0.0, min(1.0, t))
    if mode == 'cubic_out':
        return 1.0 - (1.0 - t) ** 3
    if mode == 'cubic_in':
        return t ** 3
    return t  # linear

# ── ffprobe ──────────────────────────────────────────────────────────────────
def ffprobe_video(path: Path) -> tuple[float, float]:
    """Returns (fps, duration_seconds)."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams', str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    video = next(s for s in data['streams'] if s['codec_type'] == 'video')
    num, den = map(int, video['r_frame_rate'].split('/'))
    fps = num / den
    # duration may be missing for some containers; fallback to nb_frames/fps
    duration = float(video.get('duration') or 0)
    if not duration and video.get('nb_frames'):
        duration = int(video['nb_frames']) / fps
    return fps, duration

# ── Core processing pipeline ─────────────────────────────────────────────────
def process_video(
    job_id: str,
    input_path: Path,
    split: float,
    to_peak_easing: str,
    from_peak_easing: str,
    stretch: float,
    final_fps: int,
    fast_mode: bool,
    crf: int,
):
    job_dir = WORK_DIR / job_id
    frames_dir = job_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        _job_set(job_id, status_msg=msg)

    try:
        # ── Step 1: probe ────────────────────────────────────────────────────
        log('Probing video…')
        source_fps, _ = ffprobe_video(input_path)

        # ── Step 2: extract frames with mpdecimate ───────────────────────────
        log('Extracting frames (deduplicating)…')
        subprocess.run(
            [
                'ffmpeg', '-y', '-i', str(input_path),
                '-vf', 'mpdecimate',
                '-vsync', 'vfr',
                str(frames_dir / 'frame_%06d.png'),
            ],
            capture_output=True,
            check=True,
        )

        # ── Step 3: count surviving frames ──────────────────────────────────
        frames = sorted(frames_dir.glob('frame_*.png'))
        total_frames = len(frames)
        if total_frames < 4:
            raise RuntimeError(f'Too few frames after dedup ({total_frames}). Is the input valid?')
        log(f'Got {total_frames} unique frames.')

        # ── Step 4: mid ──────────────────────────────────────────────────────
        mid = round(total_frames * split)
        mid = max(1, min(total_frames - 1, mid))

        # ── Step 5: out_fps ──────────────────────────────────────────────────
        out_fps = source_fps * stretch

        # ── Step 6: steps ────────────────────────────────────────────────────
        steps = round((mid / source_fps) * out_fps)
        steps = max(2, steps)

        # ── Step 7: build frame-index sequence ───────────────────────────────
        log('Building eased frame sequence…')
        def build_segment(n_steps: int, easing: str, start: float, end: float) -> list[int]:
            indices = []
            for i in range(n_steps):
                t = i / (n_steps - 1) if n_steps > 1 else 0.0
                val = start + ease(t, easing) * (end - start)
                idx = int(round(val))
                idx = max(0, min(total_frames - 1, idx))
                indices.append(idx)
            return indices

        seg_a = build_segment(steps, to_peak_easing, 0.0, float(mid))
        seg_b = build_segment(steps, from_peak_easing, float(mid), 0.0)
        all_indices = seg_a + seg_b

        # ── Step 8: reassemble via sequentially-numbered symlinks ────────────
        log('Assembling sequence…')
        seq_dir = job_dir / 'seq'
        seq_dir.mkdir(exist_ok=True)
        for i, idx in enumerate(all_indices):
            src = frames[idx].resolve()
            dst = seq_dir / f'{i:06d}.png'
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)

        intermediate = job_dir / 'intermediate.mp4'
        subprocess.run(
            [
                'ffmpeg', '-y',
                '-framerate', str(out_fps),
                '-i', str(seq_dir / '%06d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', str(crf), '-an',
                str(intermediate),
            ],
            capture_output=True,
            check=True,
        )

        # ── Step 9 & 10: motion interpolate → final output ───────────────────
        output_path = job_dir / 'output.mp4'
        if fast_mode:
            vf = f'fps={final_fps}'
            log('Encoding (fast mode, plain fps conversion)…')
        else:
            vf = (
                f'minterpolate=fps={final_fps}'
                ':mi_mode=mci:mc_mode=aobmc:vsbmc=1'
            )
            log('Applying motion interpolation (this may take a while)…')

        subprocess.run(
            [
                'ffmpeg', '-y',
                '-i', str(intermediate),
                '-vf', vf,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', str(crf), '-an',
                str(output_path),
            ],
            capture_output=True,
            check=True,
        )

        _job_set(job_id, status='done', output=str(output_path), status_msg='Done!')

    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode('utf-8', errors='replace')[-2000:]
        _job_set(job_id, status='error', error=f'ffmpeg error: {stderr}', status_msg='Failed.')
    except Exception as exc:
        _job_set(job_id, status='error', error=str(exc), status_msg='Failed.')
    finally:
        # clean up input & frames (keep output)
        try:
            input_path.unlink(missing_ok=True)
            shutil.rmtree(frames_dir, ignore_errors=True)
            shutil.rmtree(seq_dir, ignore_errors=True)
            intermediate = job_dir / 'intermediate.mp4'
            intermediate.unlink(missing_ok=True)
        except Exception:
            pass

# ── Periodic cleanup (files older than 1 h) ──────────────────────────────────
def _cleanup_loop():
    while True:
        time.sleep(300)  # check every 5 min
        cutoff = time.time() - 3600
        try:
            for d in WORK_DIR.iterdir():
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    with _jobs_lock:
                        _jobs.pop(d.name, None)
        except Exception:
            pass

threading.Thread(target=_cleanup_loop, daemon=True).start()

# ── Input validation helpers ─────────────────────────────────────────────────
_SAFE_ID = re.compile(r'^[a-zA-Z0-9]{1,64}$')
VALID_EASINGS = {'cubic_out', 'cubic_in', 'linear'}

def safe_float(v, lo, hi, default):
    try:
        f = float(v)
        return max(lo, min(hi, f))
    except (TypeError, ValueError):
        return default

def safe_int(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    clip = request.files.get('clip')
    if not clip or not clip.filename:
        return jsonify({'ok': False, 'error': 'No file provided.'}), 400

    ext = Path(clip.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'ok': False, 'error': f'Unsupported format: {ext}'}), 400

    # Parse params
    split        = safe_float(request.form.get('split'),        0.05, 0.95, 0.5)
    to_peak      = request.form.get('to_peak_easing',   'cubic_out')
    from_peak    = request.form.get('from_peak_easing', 'cubic_in')
    stretch      = safe_float(request.form.get('stretch'),      1.0,  6.0,  2.0)
    final_fps    = safe_int(request.form.get('final_fps'),       15,   60,   60)
    fast_mode    = request.form.get('fast_mode', 'false').lower() in ('1', 'true', 'yes')
    crf          = safe_int(request.form.get('crf'),             0,    51,   16)

    if to_peak not in VALID_EASINGS:
        to_peak = 'cubic_out'
    if from_peak not in VALID_EASINGS:
        from_peak = 'cubic_in'

    # Save upload
    job_id = uuid.uuid4().hex  # alphanumeric only
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f'input{ext}'
    clip.save(str(input_path))

    _job_set(job_id, status='processing', status_msg='Queued…', output=None, error=None)

    t = threading.Thread(
        target=process_video,
        args=(job_id, input_path, split, to_peak, from_peak, stretch, final_fps, fast_mode, crf),
        daemon=True,
    )
    t.start()

    return jsonify({'ok': True, 'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id: str):
    if not _SAFE_ID.match(job_id):
        abort(400)
    info = _job_get(job_id)
    if not info:
        return jsonify({'status': 'not_found'}), 404
    resp = {
        'status':     info.get('status', 'processing'),
        'status_msg': info.get('status_msg', ''),
    }
    if info.get('status') == 'done':
        resp['download'] = f'/result/{job_id}'
    if info.get('status') == 'error':
        resp['error'] = info.get('error', 'Unknown error.')
    return jsonify(resp)


@app.route('/result/<job_id>')
def result(job_id: str):
    if not _SAFE_ID.match(job_id):
        abort(400)
    output_path = WORK_DIR / job_id / 'output.mp4'
    if not output_path.exists():
        abort(404)
    return send_file(str(output_path), mimetype='video/mp4', as_attachment=False)


@app.route('/download/<job_id>')
def download(job_id: str):
    if not _SAFE_ID.match(job_id):
        abort(400)
    output_path = WORK_DIR / job_id / 'output.mp4'
    if not output_path.exists():
        abort(404)
    return send_file(
        str(output_path),
        mimetype='video/mp4',
        as_attachment=True,
        download_name='reverse_edit.mp4',
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
