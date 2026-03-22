"""
AudioStitcher - Automated piano recording comping tool.

Uses a "base take + patches" model: start from the best full recording,
then patch in segments from other takes only where needed.
"""

import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field
from librosa.sequence import dtw


@dataclass
class Take:
    """A single recording take."""
    path: Path
    name: str
    audio: np.ndarray  # mono, float32
    sr: int
    duration: float
    onset_env: np.ndarray = field(default=None, repr=False)
    onset_times: np.ndarray = field(default=None, repr=False)
    beat_times: np.ndarray = field(default=None, repr=False)
    chroma: np.ndarray = field(default=None, repr=False)
    tempo: float = 0.0
    # DTW warping path relative to base take
    warp_ref_frames: np.ndarray = field(default=None, repr=False)
    warp_take_frames: np.ndarray = field(default=None, repr=False)


@dataclass
class Segment:
    """A segment (typically a bar or group of bars) within the piece."""
    index: int
    start_time: float  # in base take time
    end_time: float
    label: str = ""
    cut: bool = False  # True for false starts — auto-excluded from composite


@dataclass
class Issue:
    """A detected problem in the base take for a specific segment."""
    segment_index: int
    issue_type: str   # "timing", "wrong_note", "clipping", "hesitation", "noise"
    severity: float   # 0-1, higher = worse
    description: str
    best_donor_idx: int = -1  # which other take is the best replacement, -1 if none


@dataclass
class Patch:
    """A replacement: use a different take for this segment instead of the base."""
    segment_index: int
    take_index: int  # which donor take
    reason: str = ""  # why this was patched


class Project:
    """A comping project: base take + patches from other takes."""

    def __init__(self, folder: str, sr: int = 44100):
        self.folder = Path(folder)
        self.sr = sr
        self.takes: list[Take] = []
        self.segments: list[Segment] = []
        self.patches: list[Patch] = []  # only segments that differ from base
        self.issues: list[Issue] = []  # detected problems in base take
        self.base_idx: int = 0

    def load_takes(self, min_duration: float = 30.0):
        """Load all WAV files from the project folder."""
        wav_files = sorted(self.folder.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in {self.folder}")

        print(f"Found {len(wav_files)} WAV files")
        for path in wav_files:
            y, sr = librosa.load(path, sr=self.sr, mono=True)
            duration = len(y) / sr
            if duration < min_duration:
                print(f"  Skipping {path.name} ({duration:.1f}s < {min_duration}s)")
                continue
            self.takes.append(Take(
                path=path, name=path.stem, audio=y, sr=sr, duration=duration,
            ))
            print(f"  Loaded {path.stem}: {duration:.1f}s")

        if not self.takes:
            raise ValueError("No valid takes loaded")
        print(f"\n{len(self.takes)} takes loaded")

    def set_base(self, index: int = None, name: str = None):
        """Set the base take by index or name substring."""
        if name is not None:
            matches = [i for i, t in enumerate(self.takes) if name in t.name]
            if not matches:
                raise ValueError(f"No take matching '{name}'")
            index = matches[0]
        self.base_idx = index
        print(f"Base take: {self.takes[self.base_idx].name}")

    def analyze_takes(self):
        """Compute onset envelopes, beats, tempo, and chroma for each take."""
        print("\nAnalyzing takes...")
        for take in self.takes:
            take.onset_env = librosa.onset.onset_strength(y=take.audio, sr=take.sr)
            onset_frames = librosa.onset.onset_detect(y=take.audio, sr=take.sr)
            take.onset_times = librosa.frames_to_time(onset_frames, sr=take.sr)
            tempo, beat_frames = librosa.beat.beat_track(y=take.audio, sr=take.sr)
            take.beat_times = librosa.frames_to_time(beat_frames, sr=take.sr)
            take.tempo = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            take.chroma = librosa.feature.chroma_cqt(y=take.audio, sr=take.sr)
            print(f"  {take.name}: tempo={take.tempo:.1f} BPM, "
                  f"{len(take.onset_times)} onsets, {len(take.beat_times)} beats")

    def align_takes(self):
        """Align all takes to the base using DTW on chroma features."""
        print("\nAligning takes to base via DTW...")
        base = self.takes[self.base_idx]
        n_base = base.chroma.shape[1]
        base.warp_ref_frames = np.arange(n_base)
        base.warp_take_frames = np.arange(n_base)

        for i, take in enumerate(self.takes):
            if i == self.base_idx:
                continue
            D, wp = dtw(X=base.chroma, Y=take.chroma, subseq=True)
            take.warp_ref_frames = wp[::-1, 0]
            take.warp_take_frames = wp[::-1, 1]
            mean_cost = D[wp[-1, 0], wp[-1, 1]] / len(wp)
            print(f"  {take.name}: {len(wp)} warping points, mean DTW cost={mean_cost:.4f}")

    def segment_by_bars(self, beats_per_bar: int = 4):
        """Create segments based on detected beats, including audio before first beat."""
        base = self.takes[self.base_idx]
        beats = base.beat_times
        if len(beats) == 0:
            raise ValueError("No beats detected in base take")

        self.segments = []

        # Segment 0: from start of audio to first beat (pickup/intro)
        if beats[0] > 0.1:
            self.segments.append(Segment(
                index=0, start_time=0.0, end_time=beats[0], label="Pickup",
            ))

        # Bar segments
        for i in range(0, len(beats), beats_per_bar):
            start = beats[i]
            end_idx = i + beats_per_bar
            end = beats[end_idx] if end_idx < len(beats) else base.duration
            self.segments.append(Segment(
                index=len(self.segments),
                start_time=start,
                end_time=end,
                label=f"Bar {i // beats_per_bar + 1}",
            ))

        print(f"\nCreated {len(self.segments)} segments ({beats_per_bar} beats/bar)")
        print(f"  First segment: {self.segments[0].label} "
              f"({self.segments[0].start_time:.2f}s - {self.segments[0].end_time:.2f}s)")
        print(f"  Average duration: "
              f"{np.mean([s.end_time - s.start_time for s in self.segments]):.2f}s")

    def _base_time_to_take_time(self, take: Take, base_time: float) -> float:
        """Convert a time in the base take to the corresponding time in another take.

        Uses linear interpolation on the DTW warping path for sub-frame accuracy.
        """
        if take.warp_ref_frames is None:
            return base_time
        hop_length = 512
        ref_frame = base_time * self.sr / hop_length  # float, not int
        # Interpolate: find the donor frame corresponding to this base frame
        take_frame = np.interp(ref_frame, take.warp_ref_frames, take.warp_take_frames)
        return take_frame * hop_length / self.sr

    def _refine_donor_boundary(self, take: Take, base_time: float,
                               search_ms: float = 80) -> float:
        """Refine a DTW-mapped donor time using cross-correlation with base audio.

        Takes a short window of base audio around the boundary and finds the
        best-matching position in the donor within a search range. This gives
        sample-accurate alignment, locking onto note attacks/transients.
        """
        approx_donor_time = self._base_time_to_take_time(take, base_time)
        base = self.takes[self.base_idx]

        ref_half = int(0.03 * self.sr)   # 30ms half-window for reference
        search_half = int(search_ms / 1000 * self.sr)

        # Reference window from base centered on boundary
        base_sample = int(base_time * self.sr)
        ref_start = max(0, base_sample - ref_half)
        ref_end = min(len(base.audio), base_sample + ref_half)
        ref = base.audio[ref_start:ref_end]

        if len(ref) < ref_half:
            return approx_donor_time

        # Search window from donor around DTW estimate
        donor_sample = int(approx_donor_time * self.sr)
        search_start = max(0, donor_sample - search_half)
        search_end = min(len(take.audio), donor_sample + search_half)
        search = take.audio[search_start:search_end]

        if len(search) <= len(ref):
            return approx_donor_time

        # Cross-correlation: find where ref best matches within search
        corr = np.correlate(search, ref, mode='valid')
        best_offset = np.argmax(np.abs(corr))

        # The match starts at search_start + best_offset in the donor.
        # The boundary corresponds to ref_half into the reference window.
        refined_sample = search_start + best_offset + (base_sample - ref_start)
        return refined_sample / self.sr

    def get_segment_audio(self, take_idx: int, segment: Segment,
                          match_duration: bool = True) -> np.ndarray:
        """Extract audio for a segment from a take.

        Args:
            take_idx: Which take to extract from.
            segment: The segment definition (in base take time).
            match_duration: If True and take != base, time-stretch the extracted
                audio to match the base segment's duration. This prevents tempo
                artifacts when patching.
        """
        take = self.takes[take_idx]
        target_duration = segment.end_time - segment.start_time
        target_samples = int(target_duration * self.sr)

        if take_idx == self.base_idx:
            # Direct extraction from base — no warping needed
            start_sample = max(0, int(segment.start_time * self.sr))
            end_sample = min(len(take.audio), int(segment.end_time * self.sr))
            if start_sample >= end_sample:
                return np.zeros(target_samples, dtype=np.float32)
            return take.audio[start_sample:end_sample].copy()

        # For donor takes: use cross-correlation-refined boundaries
        start = self._refine_donor_boundary(take, segment.start_time)
        end = self._refine_donor_boundary(take, segment.end_time)

        start_sample = max(0, int(start * self.sr))
        end_sample = min(len(take.audio), int(end * self.sr))

        if start_sample >= end_sample or start_sample >= len(take.audio):
            return np.zeros(target_samples, dtype=np.float32)

        audio = take.audio[start_sample:end_sample].copy()

        # Time-stretch to match base segment duration
        if match_duration and len(audio) > 0:
            donor_duration = len(audio) / self.sr
            if abs(donor_duration - target_duration) > 0.02:
                stretch_rate = donor_duration / target_duration
                audio = librosa.effects.time_stretch(audio, rate=stretch_rate)
                # Trim or pad to exact length
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                elif len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)))

        return audio

    def diagnose(self):
        """Analyze the base take for problems, find best donor for each flagged segment.

        Detects:
        - Clipping: samples hitting the rail
        - Hesitations: sudden tempo drops (local onset gap much larger than neighbors)
        - Timing irregularity: onsets deviating from the beat grid
        - Wrong notes: chroma content in the base diverging from other takes' consensus
        - Noise spikes: sudden energy bursts inconsistent with musical content
        """
        print("\nDiagnosing base take...")
        self.issues = []
        base = self.takes[self.base_idx]

        for seg in self.segments:
            seg_issues = []
            audio = self.get_segment_audio(self.base_idx, seg)
            if len(audio) < self.sr * 0.05:
                continue

            # --- False start detection ---
            # If all donor takes map to near-zero duration for this segment,
            # the base has extra material (e.g., stopped and restarted).
            false_start = False
            for ti, take in enumerate(self.takes):
                if ti == self.base_idx:
                    continue
                donor_start = self._base_time_to_take_time(take, seg.start_time)
                donor_end = self._base_time_to_take_time(take, seg.end_time)
                donor_dur = donor_end - donor_start
                base_dur = seg.end_time - seg.start_time
                if donor_dur < base_dur * 0.1:
                    false_start = True
            if false_start:
                seg_issues.append(Issue(
                    segment_index=seg.index, issue_type="false_start",
                    severity=1.0,
                    description="no matching content in donor takes (likely stopped and restarted)",
                ))

            # --- Clipping detection ---
            clip_ratio = np.mean(np.abs(audio) > 0.99)
            if clip_ratio > 0.001:
                seg_issues.append(Issue(
                    segment_index=seg.index, issue_type="clipping",
                    severity=min(1.0, clip_ratio * 100),
                    description=f"{clip_ratio*100:.2f}% samples clipped",
                ))

            # --- Hesitation detection ---
            # Look for unusually long gaps between onsets within this segment
            seg_start = seg.start_time
            seg_end = seg.end_time
            seg_onsets = base.onset_times[
                (base.onset_times >= seg_start) & (base.onset_times < seg_end)
            ]
            if len(seg_onsets) >= 3:
                gaps = np.diff(seg_onsets)
                median_gap = np.median(gaps)
                if median_gap > 0:
                    max_gap = np.max(gaps)
                    gap_ratio = max_gap / median_gap
                    # Only flag if the gap is both relatively and absolutely large
                    if gap_ratio > 4.0 and max_gap > 0.5:
                        seg_issues.append(Issue(
                            segment_index=seg.index, issue_type="hesitation",
                            severity=min(1.0, (gap_ratio - 4.0) / 6.0),
                            description=f"gap {max_gap:.2f}s ({gap_ratio:.1f}x median)",
                        ))

            # --- Timing irregularity ---
            # Compare onsets to the beat grid
            seg_beats = base.beat_times[
                (base.beat_times >= seg_start) & (base.beat_times < seg_end)
            ]
            if len(seg_onsets) >= 2 and len(seg_beats) >= 2:
                # For each onset, find distance to nearest beat
                deviations = []
                for onset in seg_onsets:
                    nearest_beat_dist = np.min(np.abs(seg_beats - onset))
                    deviations.append(nearest_beat_dist)
                mean_dev = np.mean(deviations)
                # Beat period for reference
                beat_period = np.median(np.diff(seg_beats)) if len(seg_beats) > 1 else 0.5
                dev_ratio = mean_dev / beat_period if beat_period > 0 else 0
                # Flag meaningful timing issues; rubato up to ~30% is normal
                if dev_ratio > 0.35:
                    seg_issues.append(Issue(
                        segment_index=seg.index, issue_type="timing",
                        severity=min(1.0, (dev_ratio - 0.4) / 0.3),
                        description=f"mean onset deviation {mean_dev*1000:.0f}ms "
                                    f"({dev_ratio*100:.0f}% of beat)",
                    ))

            # --- Wrong note detection (chroma consensus) ---
            if len(self.takes) >= 3:
                hop_length = 512
                start_frame = int(seg_start * self.sr / hop_length)
                end_frame = int(seg_end * self.sr / hop_length)

                base_chroma = base.chroma[:, start_frame:end_frame]
                if base_chroma.shape[1] > 0:
                    # Build consensus chroma from all other takes
                    other_chromas = []
                    for ti, take in enumerate(self.takes):
                        if ti == self.base_idx:
                            continue
                        t_start = self._base_time_to_take_time(take, seg_start)
                        t_end = self._base_time_to_take_time(take, seg_end)
                        sf_ = int(t_start * self.sr / hop_length)
                        ef_ = int(t_end * self.sr / hop_length)
                        tc = take.chroma[:, sf_:ef_]
                        if tc.shape[1] > 0:
                            # Resample to match base frame count
                            if tc.shape[1] != base_chroma.shape[1]:
                                indices = np.linspace(0, tc.shape[1] - 1,
                                                      base_chroma.shape[1]).astype(int)
                                tc = tc[:, indices]
                            other_chromas.append(tc)

                    if len(other_chromas) >= 2:
                        consensus = np.median(np.stack(other_chromas), axis=0)
                        # Cosine similarity between base and consensus per frame
                        dot = np.sum(base_chroma * consensus, axis=0)
                        norm_b = np.linalg.norm(base_chroma, axis=0) + 1e-8
                        norm_c = np.linalg.norm(consensus, axis=0) + 1e-8
                        similarity = dot / (norm_b * norm_c)
                        mean_sim = np.mean(similarity)
                        min_sim = np.min(similarity)
                        if mean_sim < 0.85 or min_sim < 0.5:
                            severity = max(1.0 - mean_sim, (0.5 - min_sim) if min_sim < 0.5 else 0)
                            seg_issues.append(Issue(
                                segment_index=seg.index, issue_type="wrong_note",
                                severity=min(1.0, severity * 2),
                                description=f"chroma deviation from consensus "
                                            f"(mean_sim={mean_sim:.2f}, min={min_sim:.2f})",
                            ))

            # --- Noise spike detection ---
            # Only flag truly abnormal spikes (not just loud chords).
            # Check that the spike frame also has high spectral flatness (noise-like).
            rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
            if len(rms) > 4:
                rms_diff = np.abs(np.diff(rms))
                median_diff = np.median(rms_diff)
                if median_diff > 0:
                    max_spike = np.max(rms_diff)
                    spike_ratio = max_spike / median_diff
                    # High threshold: piano has natural dynamics
                    if spike_ratio > 30.0:
                        seg_issues.append(Issue(
                            segment_index=seg.index, issue_type="noise",
                            severity=min(1.0, (spike_ratio - 30.0) / 200.0),
                            description=f"energy spike {spike_ratio:.1f}x median change",
                        ))

            # For each issue, find the best donor take
            for issue in seg_issues:
                issue.best_donor_idx = self._find_best_donor(seg)

            self.issues.extend(seg_issues)

        self._print_diagnosis()
        self._mark_false_starts()

    def _mark_false_starts(self):
        """Mark false start segments as cut, so they're excluded from the composite."""
        false_start_indices = {i.segment_index for i in self.issues
                               if i.issue_type == "false_start"}
        for seg in self.segments:
            seg.cut = seg.index in false_start_indices
        if false_start_indices:
            labels = [self.segments[i].label for i in sorted(false_start_indices)]
            print(f"  Marked {len(false_start_indices)} segment(s) as cut (false starts): "
                  + ", ".join(labels))

    def _find_best_donor(self, segment: Segment) -> int:
        """Find the donor take with fewest problems for this segment."""
        best_idx = -1
        best_problem_count = float('inf')

        base = self.takes[self.base_idx]

        for ti in range(len(self.takes)):
            if ti == self.base_idx:
                continue
            audio = self.get_segment_audio(ti, segment)
            if len(audio) < self.sr * 0.05:
                continue

            problems = 0
            # Quick checks on the donor
            if np.mean(np.abs(audio) > 0.99) > 0.001:
                problems += 3  # clipping is bad
            # Check onset regularity
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
            if len(onset_frames) >= 3:
                onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
                gaps = np.diff(onset_times)
                if np.median(gaps) > 0:
                    gap_ratio = np.max(gaps) / np.median(gaps)
                    if gap_ratio > 3.0:
                        problems += 1
            # Check energy stability
            rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
            if len(rms) > 4:
                rms_diff = np.abs(np.diff(rms))
                median_diff = np.median(rms_diff)
                if median_diff > 0 and np.max(rms_diff) / median_diff > 8.0:
                    problems += 1

            if problems < best_problem_count:
                best_problem_count = problems
                best_idx = ti

        return best_idx

    def _print_diagnosis(self):
        """Print a human-readable diagnostic report."""
        if not self.issues:
            print("  No issues detected in base take!")
            return

        # Group by segment
        by_segment = {}
        for issue in self.issues:
            by_segment.setdefault(issue.segment_index, []).append(issue)

        # Sort by max severity within each segment
        sorted_segs = sorted(by_segment.items(),
                             key=lambda x: max(i.severity for i in x[1]), reverse=True)

        print(f"\n  Found {len(self.issues)} issues in {len(by_segment)} segments:\n")
        print(f"  {'Segment':<12} {'Time':>12} {'Issue':<14} {'Severity':>8}  {'Description':<45} {'Best donor'}")
        print(f"  {'-'*12} {'-'*12} {'-'*14} {'-'*8}  {'-'*45} {'-'*15}")

        for seg_idx, issues in sorted_segs:
            seg = self.segments[seg_idx]
            time_str = f"{seg.start_time:.1f}-{seg.end_time:.1f}s"
            for i, issue in enumerate(issues):
                seg_label = seg.label if i == 0 else ""
                time_label = time_str if i == 0 else ""
                donor_name = self.takes[issue.best_donor_idx].name if issue.best_donor_idx >= 0 else "none"
                sev_bar = "#" * int(issue.severity * 5) + "." * (5 - int(issue.severity * 5))
                print(f"  {seg_label:<12} {time_label:>12} {issue.issue_type:<14} {sev_bar:>8}  "
                      f"{issue.description:<45} {donor_name}")

    def apply_patches_from_issues(self, issue_types: list[str] = None,
                                  min_severity: float = 0.3):
        """Convert diagnosed issues into patches.

        Args:
            issue_types: Which issue types to patch. None = all types.
            min_severity: Only patch issues at or above this severity.
        """
        self.patches = []
        seen_segments = set()

        for issue in self.issues:
            if issue.best_donor_idx < 0:
                continue
            if issue.severity < min_severity:
                continue
            if issue_types and issue.issue_type not in issue_types:
                continue
            if issue.segment_index in seen_segments:
                continue
            seen_segments.add(issue.segment_index)
            self.patches.append(Patch(
                segment_index=issue.segment_index,
                take_index=issue.best_donor_idx,
                reason=f"{issue.issue_type}: {issue.description}",
            ))

        print(f"\n{len(self.patches)} patches queued from {len(self.issues)} diagnosed issues "
              f"(min_severity={min_severity}):")
        for p in self.patches:
            seg = self.segments[p.segment_index]
            donor = self.takes[p.take_index]
            print(f"  {seg.label} ({seg.start_time:.1f}-{seg.end_time:.1f}s): "
                  f"{donor.name} — {p.reason}")

    def add_patch(self, segment_index: int, take_index: int, reason: str = "manual"):
        """Manually add a patch."""
        self.patches = [p for p in self.patches if p.segment_index != segment_index]
        self.patches.append(Patch(segment_index, take_index, reason))
        seg = self.segments[segment_index]
        donor = self.takes[take_index]
        print(f"Patched {seg.label} ({seg.start_time:.1f}-{seg.end_time:.1f}s) "
              f"with {donor.name}")

    def remove_patch(self, segment_index: int):
        """Remove a patch, reverting to base take."""
        self.patches = [p for p in self.patches if p.segment_index != segment_index]
        seg = self.segments[segment_index]
        print(f"Reverted {seg.label} to base take")

    def _is_false_start_segment(self, seg_index: int) -> bool:
        """Check if a segment has a false_start issue."""
        return any(i.segment_index == seg_index and i.issue_type == "false_start"
                   for i in self.issues)

    def realign_takes(self):
        """Re-align takes using cleaned base chroma (false start regions removed).

        After diagnose() identifies false starts, the DTW warping paths are
        distorted near those regions because the base has extra material that
        doesn't exist in donor takes. This method re-runs DTW with the false
        start frames excluded, giving accurate alignment for surrounding segments.
        """
        false_start_segs = [seg for seg in self.segments if seg.cut]
        if not false_start_segs:
            return  # No false starts, alignment is fine

        print("\nRe-aligning takes with false start regions excluded...")
        base = self.takes[self.base_idx]
        hop_length = 512
        n_frames = base.chroma.shape[1]

        # Build mask: True = keep, False = false start region
        keep_mask = np.ones(n_frames, dtype=bool)
        for seg in false_start_segs:
            start_frame = int(seg.start_time * self.sr / hop_length)
            end_frame = int(seg.end_time * self.sr / hop_length)
            keep_mask[start_frame:min(end_frame, n_frames)] = False

        n_removed = np.sum(~keep_mask)
        print(f"  Excluded {n_removed} frames ({n_removed * hop_length / self.sr:.2f}s) "
              f"from {len(false_start_segs)} false start segment(s)")

        clean_chroma = base.chroma[:, keep_mask]
        # Mapping from clean frame index back to original base frame index
        clean_to_orig = np.where(keep_mask)[0]

        for i, take in enumerate(self.takes):
            if i == self.base_idx:
                continue
            D, wp = dtw(X=clean_chroma, Y=take.chroma, subseq=True)
            clean_ref = wp[::-1, 0]
            take_frames = wp[::-1, 1]

            # Convert clean-space ref frames back to original base frame space
            take.warp_ref_frames = clean_to_orig[clean_ref]
            take.warp_take_frames = take_frames

            mean_cost = D[wp[-1, 0], wp[-1, 1]] / len(wp)
            print(f"  {take.name}: {len(wp)} warping points, mean DTW cost={mean_cost:.4f}")

    def export_composite(self, output_path: str = None, crossfade_ms: int = 50) -> np.ndarray:
        """Export the composite: base take with patches applied.

        False start patches are cut out entirely. Regular patches replace
        the base audio with volume-matched donor audio.

        Crossfades happen in the pre-boundary sustain zone: the incoming
        source provides a short pre-roll before the segment boundary, which
        is blended with the outgoing source's tail. This ensures note attacks
        at segment boundaries come through cleanly without doubling.

        Returns the composite audio array. If output_path is given, also writes to file.
        """
        base = self.takes[self.base_idx]
        crossfade_samples = int(crossfade_ms / 1000 * self.sr)

        patched_indices = {p.segment_index: p.take_index for p in self.patches}
        cut_indices = {seg.index for seg in self.segments if seg.cut}

        # Group consecutive segments by source
        groups = []  # list of (source, [seg, seg, ...])
        for seg in self.segments:
            if seg.index in cut_indices:
                groups.append(('cut', [seg]))
            elif seg.index in patched_indices:
                donor_idx = patched_indices[seg.index]
                if groups and groups[-1][0] == donor_idx:
                    groups[-1][1].append(seg)
                else:
                    groups.append((donor_idx, [seg]))
            else:
                if groups and groups[-1][0] == 'base':
                    groups[-1][1].append(seg)
                else:
                    groups.append(('base', [seg]))

        # Extract audio for each group: core (exact segment range) + pre-roll
        # for crossfading. The pre-roll comes from the INCOMING source and
        # covers the sustain zone just before the boundary, so the crossfade
        # blends two versions of the same pre-boundary region rather than
        # mixing post-boundary with pre-boundary audio (which causes doubled
        # note attacks).
        runs = []  # list of (source, core_audio, preroll_audio_or_None)

        non_cut_count = 0
        for source, segs in groups:
            if source == 'cut':
                runs.append(('cut', None, None))
                continue

            first_start = segs[0].start_time
            last_end = segs[-1].end_time
            need_preroll = non_cut_count > 0

            if source == 'base':
                s = max(0, int(first_start * self.sr))
                e = min(len(base.audio), int(last_end * self.sr))
                core = base.audio[s:e].copy()

                preroll = None
                if need_preroll:
                    ps = max(0, s - crossfade_samples)
                    preroll = base.audio[ps:s].copy()

                runs.append(('base', core, preroll))
            else:
                donor_idx = source
                take = self.takes[donor_idx]
                donor_start = self._refine_donor_boundary(take, first_start)
                donor_end = self._refine_donor_boundary(take, last_end)
                cs = max(0, int(donor_start * self.sr))
                ce = min(len(take.audio), int(donor_end * self.sr))

                if cs < ce:
                    core = take.audio[cs:ce].copy()

                    # Volume-match to base
                    bs = int(first_start * self.sr)
                    be = min(int(last_end * self.sr), len(base.audio))
                    base_rms = np.sqrt(np.mean(base.audio[bs:be]**2)) + 1e-8
                    core_rms = np.sqrt(np.mean(core**2)) + 1e-8
                    gain = base_rms / core_rms
                    core *= gain

                    preroll = None
                    if need_preroll:
                        pre_time = max(0, first_start - crossfade_samples / self.sr)
                        donor_pre = self._refine_donor_boundary(take, pre_time)
                        ps = max(0, int(donor_pre * self.sr))
                        preroll = take.audio[ps:cs].copy() * gain

                    runs.append((donor_idx, core, preroll))
                else:
                    runs.append((donor_idx, np.zeros(0, dtype=np.float32), None))

            non_cut_count += 1

        # Concatenate runs with pre-boundary crossfades
        active = [(s, c, p) for s, c, p in runs
                   if s != 'cut' and c is not None and len(c) > 0]

        if not active:
            output = np.zeros(0, dtype=np.float32)
        else:
            output = active[0][1]
            for i in range(1, len(active)):
                _, core, preroll = active[i]

                if preroll is not None and len(preroll) > 0:
                    # Crossfade in the pre-boundary sustain zone
                    cf = min(len(preroll), len(output))
                    if cf > 1:
                        pr = preroll[-cf:] if len(preroll) > cf else preroll
                        cf = len(pr)

                        t = np.linspace(0, np.pi / 2, cf, dtype=np.float32)
                        fade_out = np.cos(t)
                        fade_in = np.sin(t)

                        # Both output[-cf:] and pr cover the same pre-boundary
                        # time region, so the crossfade blends sustain with
                        # sustain. The core starts at the boundary with its
                        # note attack fully preserved.
                        output[-cf:] = output[-cf:] * fade_out + pr * fade_in
                        output = np.concatenate([output, core])
                    else:
                        output = np.concatenate([output, core])
                else:
                    output = np.concatenate([output, core])

        peak = np.max(np.abs(output)) if len(output) > 0 else 0
        if peak > 0.95:
            output *= 0.95 / peak

        n_cut = len(cut_indices)
        n_patched = len(self.patches) - n_cut
        n_base = len(self.segments) - len(self.patches)

        if output_path:
            sf.write(output_path, output, self.sr)
            print(f"\nExported composite: {output_path} ({len(output)/self.sr:.1f}s)")
        print(f"  {n_base} from base, {n_patched} patched, {n_cut} cut")

        return output

    def export_segment_comparisons(self, output_dir: str, segments: list[int] = None):
        """Export specific segments from all takes for A/B listening.

        Args:
            output_dir: Directory to write WAV files.
            segments: List of segment indices to export. None = all.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        target_segs = self.segments if segments is None else [
            s for s in self.segments if s.index in segments
        ]

        for seg in target_segs:
            for ti, take in enumerate(self.takes):
                audio = self.get_segment_audio(ti, seg, match_duration=True)
                if len(audio) < self.sr * 0.05:
                    continue
                marker = "_BASE" if ti == self.base_idx else ""
                filename = out / f"seg{seg.index:03d}_{seg.label}_take{ti}_{take.name}{marker}.wav"
                sf.write(str(filename), audio, self.sr)

        print(f"\nExported {len(target_segs)} segment comparisons to {output_dir}")

    def save_session(self, path: str):
        """Save project state to JSON."""
        data = {
            "folder": str(self.folder),
            "base_idx": self.base_idx,
            "takes": [{"name": t.name, "tempo": float(t.tempo)} for t in self.takes],
            "segments": [{"index": s.index, "start": float(s.start_time),
                          "end": float(s.end_time), "label": s.label}
                         for s in self.segments],
            "patches": [{"segment": p.segment_index, "take": p.take_index,
                          "reason": p.reason}
                         for p in self.patches],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Session saved to {path}")

    def load_session(self, path: str):
        """Load session from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.base_idx = data["base_idx"]
        self.segments = [Segment(s["index"], s["start"], s["end"], s["label"])
                         for s in data["segments"]]
        self.patches = [Patch(p["segment"], p["take"], p.get("reason", ""))
                        for p in data["patches"]]
        print(f"Session loaded from {path}")


def run_project(folder: str, base_name: str = None, beats_per_bar: int = 4,
                min_duration: float = 30.0):
    """Run the full comping pipeline: load, analyze, align, diagnose.

    Returns a Project ready for review. Typical workflow after this:
        proj = run_project(...)
        # Review the diagnosis report printed above
        proj.apply_patches_from_issues(min_severity=0.5)  # auto-patch severe issues
        # Or manually: proj.add_patch(segment_index=5, take_index=2)
        proj.export_composite('output.wav')
    """
    proj = Project(folder)
    proj.load_takes(min_duration=min_duration)
    if base_name:
        proj.set_base(name=base_name)
    else:
        proj.base_idx = max(range(len(proj.takes)), key=lambda i: proj.takes[i].duration)
        print(f"Base take (longest): {proj.takes[proj.base_idx].name}")
    proj.analyze_takes()
    proj.align_takes()
    proj.segment_by_bars(beats_per_bar=beats_per_bar)
    proj.diagnose()
    proj.realign_takes()
    return proj
