"""
AudioStitcher Web UI — review diagnosed segments, toggle patches, export composite.
"""

import io
import json
import numpy as np
import soundfile as sf
from flask import Flask, render_template, jsonify, request, send_file
from audiostitcher import Project

app = Flask(__name__)
proj: Project = None


def init_project(folder: str, base_name: str = None, beats_per_bar: int = 4,
                 min_duration: float = 30.0, take_names: list[str] = None):
    """Initialize the project and run the analysis pipeline."""
    global proj
    proj = Project(folder)
    proj.load_takes(min_duration=min_duration)
    if take_names:
        # Always include the base take in the filter
        if base_name and base_name not in take_names:
            take_names.append(base_name)
        proj.takes = [t for t in proj.takes if any(n in t.name for n in take_names)]
    if base_name:
        proj.set_base(name=base_name)
    else:
        proj.base_idx = max(range(len(proj.takes)), key=lambda i: proj.takes[i].duration)
    proj.analyze_takes()
    proj.align_takes()
    proj.segment_by_bars(beats_per_bar=beats_per_bar)
    proj.diagnose()
    proj.realign_takes()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/project")
def get_project():
    """Return full project state for the UI."""
    # Build issues lookup by segment
    issues_by_seg = {}
    for issue in proj.issues:
        issues_by_seg.setdefault(issue.segment_index, []).append({
            "type": issue.issue_type,
            "severity": float(issue.severity),
            "description": issue.description,
            "best_donor_idx": issue.best_donor_idx,
        })

    # Build patches lookup
    patches_by_seg = {}
    for p in proj.patches:
        patches_by_seg[p.segment_index] = {
            "take_index": p.take_index,
            "reason": p.reason,
        }

    segments = []
    for seg in proj.segments:
        seg_issues = issues_by_seg.get(seg.index, [])
        max_severity = max((i["severity"] for i in seg_issues), default=0.0)
        patch = patches_by_seg.get(seg.index)
        segments.append({
            "index": seg.index,
            "label": seg.label,
            "start": float(seg.start_time),
            "end": float(seg.end_time),
            "duration": float(seg.end_time - seg.start_time),
            "issues": seg_issues,
            "max_severity": float(max_severity),
            "patch": patch,
            "cut": seg.cut,
            "speed_adjust": proj.speed_adjustments.get(seg.index, 0.0),
        })

    takes = [{"index": i, "name": t.name, "duration": float(t.duration),
              "tempo": float(t.tempo), "is_base": i == proj.base_idx}
             for i, t in enumerate(proj.takes)]

    return jsonify({
        "takes": takes,
        "base_idx": proj.base_idx,
        "segments": segments,
        "total_duration": float(proj.takes[proj.base_idx].duration),
    })


@app.route("/api/audio/segment/<int:seg_idx>/<int:take_idx>")
def get_segment_audio(seg_idx, take_idx):
    """Serve a segment's audio as WAV for playback."""
    seg = proj.segments[seg_idx]
    audio = proj.get_segment_audio(take_idx, seg, match_duration=(take_idx != proj.base_idx))

    # If requesting a donor, optionally volume-match to base
    if take_idx != proj.base_idx:
        base_audio = proj.get_segment_audio(proj.base_idx, seg)
        base_rms = np.sqrt(np.mean(base_audio**2)) + 1e-8
        donor_rms = np.sqrt(np.mean(audio**2)) + 1e-8
        audio = audio * (base_rms / donor_rms)

    buf = io.BytesIO()
    sf.write(buf, audio, proj.sr, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav')


@app.route("/api/audio/full/<int:take_idx>")
def get_full_take(take_idx):
    """Serve a full take's audio as WAV."""
    take = proj.takes[take_idx]
    buf = io.BytesIO()
    sf.write(buf, take.audio, take.sr, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav')


@app.route("/api/audio/range/<int:start_seg>/<int:end_seg>/<int:take_idx>")
def get_range_audio(start_seg, end_seg, take_idx):
    """Serve audio for a contiguous range of segments from a take."""
    segs = [s for s in proj.segments if start_seg <= s.index <= end_seg]
    if not segs:
        return "No segments", 404

    if take_idx == proj.base_idx:
        # Direct extraction from base
        start_sample = int(segs[0].start_time * proj.sr)
        end_sample = min(int(segs[-1].end_time * proj.sr), len(proj.takes[proj.base_idx].audio))
        audio = proj.takes[proj.base_idx].audio[start_sample:end_sample].copy()
    else:
        # Extract contiguous block from donor using DTW endpoints
        take = proj.takes[take_idx]
        donor_start = proj._refine_donor_boundary(take, segs[0].start_time)
        donor_end = proj._refine_donor_boundary(take, segs[-1].end_time)
        start_sample = max(0, int(donor_start * proj.sr))
        end_sample = min(len(take.audio), int(donor_end * proj.sr))
        if start_sample >= end_sample:
            audio = np.zeros(1000, dtype=np.float32)
        else:
            audio = take.audio[start_sample:end_sample].copy()
            # Time-stretch to match base duration
            target_duration = segs[-1].end_time - segs[0].start_time
            donor_duration = len(audio) / proj.sr
            if abs(donor_duration - target_duration) > 0.02:
                stretch_rate = donor_duration / target_duration
                import librosa
                audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
                target_samples = int(target_duration * proj.sr)
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                elif len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)))
            # Volume-match to base span
            base_start = int(segs[0].start_time * proj.sr)
            base_end = min(int(segs[-1].end_time * proj.sr), len(proj.takes[proj.base_idx].audio))
            base_span = proj.takes[proj.base_idx].audio[base_start:base_end]
            base_rms = np.sqrt(np.mean(base_span**2)) + 1e-8
            donor_rms = np.sqrt(np.mean(audio**2)) + 1e-8
            audio = audio * (base_rms / donor_rms)

    buf = io.BytesIO()
    sf.write(buf, audio, proj.sr, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav')


@app.route("/api/patch_range", methods=["POST"])
def set_patch_range():
    """Add or remove patches for a range of segments."""
    data = request.json
    start_seg = data["start_segment"]
    end_seg = data["end_segment"]
    # Skip cut segments (false starts)
    cut_indices = {s.index for s in proj.segments if s.cut}
    if data.get("remove"):
        for idx in range(start_seg, end_seg + 1):
            if idx not in cut_indices:
                proj.remove_patch(idx)
    else:
        take_idx = data["take_index"]
        reason = data.get("reason", "manual (range)")
        for idx in range(start_seg, end_seg + 1):
            if idx not in cut_indices:
                proj.add_patch(idx, take_idx, reason)
    return jsonify({"ok": True})


@app.route("/api/cut", methods=["POST"])
def toggle_cut():
    """Toggle a segment's cut flag."""
    data = request.json
    seg_idx = data["segment_index"]
    seg = proj.segments[seg_idx]
    seg.cut = data.get("cut", not seg.cut)
    return jsonify({"ok": True, "cut": seg.cut})


@app.route("/api/speed", methods=["POST"])
def set_speed():
    """Set speed adjustment for a range of segments."""
    data = request.json
    start_seg = data["start_segment"]
    end_seg = data["end_segment"]
    speed_pct = float(data["speed"])  # percent change, e.g. -2.0 = 2% slower
    for idx in range(start_seg, end_seg + 1):
        seg = proj.segments[idx]
        if seg.cut:
            continue
        if speed_pct == 0.0:
            proj.speed_adjustments.pop(idx, None)
        else:
            proj.speed_adjustments[idx] = speed_pct
    return jsonify({"ok": True})


@app.route("/api/patch", methods=["POST"])
def set_patch():
    """Add or remove a patch."""
    data = request.json
    seg_idx = data["segment_index"]
    if data.get("remove"):
        proj.remove_patch(seg_idx)
    else:
        take_idx = data["take_index"]
        reason = data.get("reason", "manual")
        proj.add_patch(seg_idx, take_idx, reason)
    return jsonify({"ok": True})


@app.route("/api/auto_patch", methods=["POST"])
def auto_patch():
    """Apply patches from issues with given threshold."""
    data = request.json
    min_severity = data.get("min_severity", 0.3)
    proj.apply_patches_from_issues(min_severity=min_severity)
    return jsonify({"ok": True, "count": len(proj.patches)})


@app.route("/api/export", methods=["POST"])
def export_composite():
    """Export the composite and return it as a WAV download."""
    data = request.json or {}
    crossfade_ms = data.get("crossfade_ms", 80)

    output = proj.export_composite(crossfade_ms=crossfade_ms)

    buf = io.BytesIO()
    sf.write(buf, output, proj.sr, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav',
                     as_attachment=True, download_name='composite.wav')


if __name__ == "__main__":
    import sys
    from waitress import serve

    folder = sys.argv[1] if len(sys.argv) > 1 else '../Projets/20250517 - Etude en fa mineur - Mendelssohn'
    base = sys.argv[2] if len(sys.argv) > 2 else '0961'
    takes = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    init_project(folder, base_name=base, beats_per_bar=4, take_names=takes)
    print("\n--- Starting web UI at http://localhost:5000 ---\n")
    serve(app, host='127.0.0.1', port=5000)
