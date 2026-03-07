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
