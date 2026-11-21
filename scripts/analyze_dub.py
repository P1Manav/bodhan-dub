#!/usr/bin/env python3
"""
scripts/analyze_dub.py

Automated reference video and audio dubbing analyzer.
Uses ONLY ffmpeg/ffprobe, numpy, and pyloudnorm (NO ASR models).
Evaluates container alignment, loudness, timing/onset sync, audio leakage, and coverage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyloudnorm as pyln


def probe_container(file_path: Path) -> Dict[str, Any]:
    """Extracts container and stream metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    video_dur = float(video_streams[0]["duration"]) if video_streams and "duration" in video_streams[0] else None
    audio_dur = float(audio_streams[0]["duration"]) if audio_streams and "duration" in audio_streams[0] else None

    fmt_dur = float(fmt.get("duration", 0.0))
    fmt_start = float(fmt.get("start_time", 0.0))

    format_tags = fmt.get("tags", {})
    audio_tags = audio_streams[0].get("tags", {}) if audio_streams else {}
    has_ai_comment = False
    for t_dict in [format_tags, audio_tags]:
        for _, v in t_dict.items():
            if "ai-dubbed" in str(v).lower() or "indic-speak" in str(v).lower():
                has_ai_comment = True

    return {
        "file": str(file_path),
        "format_duration_s": fmt_dur,
        "format_start_time_s": fmt_start,
        "video_codec": video_streams[0].get("codec_name") if video_streams else None,
        "video_duration_s": video_dur,
        "audio_codec": audio_streams[0].get("codec_name") if audio_streams else None,
        "audio_duration_s": audio_dur,
        "audio_sample_rate": int(audio_streams[0].get("sample_rate", 0)) if audio_streams else None,
        "audio_channels": int(audio_streams[0].get("channels", 0)) if audio_streams else None,
        "audio_lang": audio_tags.get("language"),
        "has_ai_dubbed_tag": has_ai_comment,
        "format_tags": format_tags,
        "audio_tags": audio_tags,
    }


def extract_audio(file_path: Path, sr: int = 48000) -> np.ndarray:
    """Decodes audio to 48kHz mono float32 array using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(file_path),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def compute_true_peak(audio: np.ndarray, oversample: int = 4) -> float:
    """Computes true peak in dBTP using oversampling interpolation."""
    if len(audio) == 0:
        return -100.0
    try:
        from scipy import signal
        oversampled = signal.resample_poly(audio, oversample, 1)
        max_val = np.max(np.abs(oversampled))
    except Exception:
        max_val = np.max(np.abs(audio))
    return float(20 * np.log10(max(1e-9, max_val)))


def compute_loudness(audio: np.ndarray, sr: int = 48000) -> Dict[str, float]:
    """Computes integrated LUFS, Loudness Range (LRA approximation), and true peak."""
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(audio))
    except Exception:
        lufs = -100.0

    peak_dbtp = compute_true_peak(audio, oversample=4)

    win_len = 3 * sr
    hop = sr
    st_loudness = []
    for i in range(0, len(audio) - win_len, hop):
        w = audio[i : i + win_len]
        try:
            val = meter.integrated_loudness(w)
            if val > -70.0:
                st_loudness.append(val)
        except Exception:
            pass

    if len(st_loudness) >= 2:
        lra = float(np.percentile(st_loudness, 95) - np.percentile(st_loudness, 10))
    else:
        lra = 0.0

    return {
        "integrated_lufs": round(lufs, 2),
        "lra_lu": round(lra, 2),
        "true_peak_dbtp": round(peak_dbtp, 2),
    }


def compute_energy_vad_regions(
    audio: np.ndarray,
    sr: int = 48000,
    frame_ms: int = 100,
    th_db: float = -37.0,
    min_speech_ms: int = 200,
    min_silence_ms: int = 450,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Extracts speech regions and binary 100ms speech mask using energy VAD."""
    frame_size = int((frame_ms / 1000.0) * sr)
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return [], np.zeros(0, dtype=bool)

    rms = np.array([np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2)) for i in range(n_frames)])
    db = 20 * np.log10(np.maximum(1e-9, rms))
    raw_speech = db > th_db

    min_sil_frames = max(1, int(min_silence_ms / frame_ms))
    min_spk_frames = max(1, int(min_speech_ms / frame_ms))

    smoothed = raw_speech.copy()
    sil_count = 0
    for i in range(len(smoothed)):
        if not smoothed[i]:
            sil_count += 1
        else:
            if 0 < sil_count <= min_sil_frames:
                smoothed[i - sil_count : i] = True
            sil_count = 0

    regions: List[Tuple[float, float]] = []
    in_speech = False
    start = 0
    for i in range(len(smoothed)):
        if smoothed[i] and not in_speech:
            in_speech = True
            start = i
        elif not smoothed[i] and in_speech:
            in_speech = False
            if (i - start) >= min_spk_frames:
                regions.append((round(start * (frame_ms / 1000.0), 3), round(i * (frame_ms / 1000.0), 3)))

    if in_speech and (len(smoothed) - start) >= min_spk_frames:
        regions.append((round(start * (frame_ms / 1000.0), 3), round(len(smoothed) * (frame_ms / 1000.0), 3)))

    return regions, smoothed


def analyze_waveform_leakage(
    source_audio: np.ndarray,
    dubbed_audio: np.ndarray,
    sr: int = 48000,
    skipped_intervals: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """Per 1s window waveform correlation between source and dubbed tracks."""
    skipped_intervals = skipped_intervals or []
    min_len = min(len(source_audio), len(dubbed_audio))
    win_size = sr
    high_corr_windows = []
    unexplained_leakage = []

    for t in range(0, min_len - win_size, win_size):
        sec = round(t / sr, 2)
        s_win = source_audio[t : t + win_size]
        d_win = dubbed_audio[t : t + win_size]
        std_s = np.std(s_win)
        std_d = np.std(d_win)
        if std_s > 1e-4 and std_d > 1e-4:
            corr = float(np.corrcoef(s_win, d_win)[0, 1])
            if corr > 0.8:
                entry = {"start_s": sec, "end_s": round(sec + 1.0, 2), "correlation": round(corr, 3)}
                high_corr_windows.append(entry)
                is_expected = False
                for sk_start, sk_end in skipped_intervals:
                    if not (sec + 1.0 < sk_start or sec > sk_end):
                        is_expected = True
                        break
                if not is_expected:
                    unexplained_leakage.append(entry)

    return {
        "total_high_correlation_windows": len(high_corr_windows),
        "high_correlation_windows": high_corr_windows,
        "unexplained_leakage_count": len(unexplained_leakage),
        "unexplained_leakage_windows": unexplained_leakage,
        "mute_bug_detected": len(unexplained_leakage) > 0,
    }


def analyze_silence_and_coverage(
    src_mask: np.ndarray,
    dub_mask: np.ndarray,
    frame_ms: int = 100,
    silence_threshold_s: float = 3.0,
) -> Dict[str, Any]:
    """Detects long silences (> 3s) in dubbed track where source audio has active speech."""
    min_len = min(len(src_mask), len(dub_mask))
    src = src_mask[:min_len]
    dub = dub_mask[:min_len]

    dropped_frames = src & ~dub
    min_frames = int(silence_threshold_s / (frame_ms / 1000.0))

    dropped_regions = []
    in_drop = False
    start = 0
    for i in range(len(dropped_frames)):
        if dropped_frames[i] and not in_drop:
            in_drop = True
            start = i
        elif not dropped_frames[i] and in_drop:
            in_drop = False
            if (i - start) >= min_frames:
                dropped_regions.append({
                    "start_s": round(start * (frame_ms / 1000.0), 2),
                    "end_s": round(i * (frame_ms / 1000.0), 2),
                    "duration_s": round((i - start) * (frame_ms / 1000.0), 2),
                })
    if in_drop and (len(dropped_frames) - start) >= min_frames:
        dropped_regions.append({
            "start_s": round(start * (frame_ms / 1000.0), 2),
            "end_s": round(len(dropped_frames) * (frame_ms / 1000.0), 2),
            "duration_s": round((len(dropped_frames) - start) * (frame_ms / 1000.0), 2),
        })

    return {
        "dropped_speech_regions_count": len(dropped_regions),
        "dropped_speech_regions": dropped_regions,
    }


def compare_with_report(report_path: Path, measured: Dict[str, Any]) -> Dict[str, Any]:
    """Compares measured values with claimed stats in report.json."""
    if not report_path.exists():
        return {"error": f"Report file not found: {report_path}"}

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except Exception as e:
        return {"error": f"Failed to read report.json: {e}"}

    hi_stats = rep.get("languages", {}).get("hi", {})
    claimed_sync_max_ms = hi_stats.get("sync_offset_stats", {}).get("max_ms")
    claimed_sync_p95_ms = hi_stats.get("sync_offset_stats", {}).get("p95_ms")
    claimed_lufs = hi_stats.get("integrated_lufs")
    claimed_dur_delta_ms = hi_stats.get("duration_delta_ms")

    meas_sync_max_ms = measured["timing_alignment"]["max_abs_offset_ms"]
    meas_sync_p95_ms = measured["timing_alignment"]["p95_abs_offset_ms"]
    meas_lufs = measured["loudness"]["dubbed"]["integrated_lufs"]
    meas_dur_delta_ms = measured["container"]["duration_delta_ms"]

    mismatches = []
    if claimed_sync_max_ms is not None and meas_sync_max_ms > (claimed_sync_max_ms + 100.0):
        mismatches.append({
            "metric": "sync_max_ms",
            "claimed": claimed_sync_max_ms,
            "measured": meas_sync_max_ms,
            "issue": f"Measured max sync offset ({meas_sync_max_ms:.1f}ms) is significantly worse than claimed ({claimed_sync_max_ms:.1f}ms)",
        })

    if claimed_sync_p95_ms is not None and meas_sync_p95_ms > (claimed_sync_p95_ms + 50.0):
        mismatches.append({
            "metric": "sync_p95_ms",
            "claimed": claimed_sync_p95_ms,
            "measured": meas_sync_p95_ms,
            "issue": f"Measured p95 sync offset ({meas_sync_p95_ms:.1f}ms) is worse than claimed ({claimed_sync_p95_ms:.1f}ms)",
        })

    if claimed_lufs is not None and abs(meas_lufs - claimed_lufs) > 1.5:
        mismatches.append({
            "metric": "integrated_lufs",
            "claimed": claimed_lufs,
            "measured": meas_lufs,
            "issue": f"Measured LUFS ({meas_lufs:.2f}) differs from claimed ({claimed_lufs:.2f})",
        })

    if claimed_dur_delta_ms is not None and abs(meas_dur_delta_ms - claimed_dur_delta_ms) > 100.0:
        mismatches.append({
            "metric": "duration_delta_ms",
            "claimed": claimed_dur_delta_ms,
            "measured": meas_dur_delta_ms,
            "issue": f"Measured duration delta ({meas_dur_delta_ms:.1f}ms) differs from claimed ({claimed_dur_delta_ms:.1f}ms)",
        })

    return {
        "claimed_stats": {
            "sync_max_ms": claimed_sync_max_ms,
            "sync_p95_ms": claimed_sync_p95_ms,
            "integrated_lufs": claimed_lufs,
            "duration_delta_ms": claimed_dur_delta_ms,
        },
        "mismatches": mismatches,
        "is_report_honest": len(mismatches) == 0,
    }


def analyze_dub(
    source_path: Path,
    dubbed_path: Path,
    report_path: Optional[Path] = None,
    segments_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs full dubbing analysis."""
    src_cont = probe_container(source_path)
    dub_cont = probe_container(dubbed_path)
    dur_delta_ms = round(abs(dub_cont["format_duration_s"] - src_cont["format_duration_s"]) * 1000.0, 2)
    container_pass = dur_delta_ms <= 100.0

    src_audio = extract_audio(source_path, sr=48000)
    dub_audio = extract_audio(dubbed_path, sr=48000)

    src_loud = compute_loudness(src_audio, sr=48000)
    dub_loud = compute_loudness(dub_audio, sr=48000)
    loudness_pass = (-17.0 <= dub_loud["integrated_lufs"] <= -15.0) and (dub_loud["true_peak_dbtp"] <= -1.0)

    src_regions, src_mask = compute_energy_vad_regions(src_audio, sr=48000)
    dub_regions, dub_mask = compute_energy_vad_regions(dub_audio, sr=48000)

    min_frames = min(len(src_mask), len(dub_mask))
    s_m = src_mask[:min_frames]
    d_m = dub_mask[:min_frames]

    dubbed_speech_source_silent_s = round(float(np.sum(d_m & ~s_m) * 0.1), 2)
    source_speech_dubbed_silent_s = round(float(np.sum(s_m & ~d_m) * 0.1), 2)

    src_leading_gap_s = src_regions[0][0] if src_regions else 0.0
    dub_leading_gap_s = dub_regions[0][0] if dub_regions else 0.0
    src_trailing_gap_s = round(src_cont["format_duration_s"] - (src_regions[-1][1] if src_regions else 0.0), 2)
    dub_trailing_gap_s = round(dub_cont["format_duration_s"] - (dub_regions[-1][1] if dub_regions else 0.0), 2)

    per_region_offsets_ms = []
    regions_beyond_300ms = 0
    for s_start, _ in src_regions:
        if not dub_regions:
            break
        closest_d = min(dub_regions, key=lambda r: abs(r[0] - s_start))
        offset_ms = round((closest_d[0] - s_start) * 1000.0, 1)
        per_region_offsets_ms.append({
            "source_onset_s": s_start,
            "dubbed_onset_s": closest_d[0],
            "offset_ms": offset_ms,
        })
        if abs(offset_ms) > 300.0:
            regions_beyond_300ms += 1

    abs_offsets = [abs(x["offset_ms"]) for x in per_region_offsets_ms]
    max_abs_offset = float(np.max(abs_offsets)) if abs_offsets else 0.0
    p95_abs_offset = float(np.percentile(abs_offsets, 95)) if abs_offsets else 0.0

    skipped_intervals = []
    if segments_path and segments_path.exists():
        try:
            with open(segments_path, "r", encoding="utf-8") as f:
                seg_data = json.load(f)
                for s in seg_data:
                    if s.get("status") in ["skipped", "translation_failed"] or s.get("is_skipped"):
                        skipped_intervals.append((float(s.get("start", 0)), float(s.get("end", 0))))
        except Exception:
            pass

    leakage_analysis = analyze_waveform_leakage(src_audio, dub_audio, sr=48000, skipped_intervals=skipped_intervals)
    coverage_analysis = analyze_silence_and_coverage(src_mask, dub_mask, frame_ms=100, silence_threshold_s=3.0)

    result: Dict[str, Any] = {
        "container": {
            "pass": container_pass,
            "duration_delta_ms": dur_delta_ms,
            "source": src_cont,
            "dubbed": dub_cont,
        },
        "loudness": {
            "pass": loudness_pass,
            "target": "-16 +/- 1 LUFS, True Peak <= -1.0 dBTP",
            "source": src_loud,
            "dubbed": dub_loud,
        },
        "timing_alignment": {
            "source_regions_count": len(src_regions),
            "dubbed_regions_count": len(dub_regions),
            "regions_beyond_300ms_count": regions_beyond_300ms,
            "max_abs_offset_ms": round(max_abs_offset, 1),
            "p95_abs_offset_ms": round(p95_abs_offset, 1),
            "leading_speech_gap_s": {
                "source": src_leading_gap_s,
                "dubbed": dub_leading_gap_s,
            },
            "trailing_speech_gap_s": {
                "source": src_trailing_gap_s,
                "dubbed": dub_trailing_gap_s,
            },
            "speech_mismatch_s": {
                "dubbed_speech_source_silent_s": dubbed_speech_source_silent_s,
                "source_speech_dubbed_silent_s": source_speech_dubbed_silent_s,
            },
            "per_region_offsets_ms": per_region_offsets_ms,
        },
        "original_audio_leakage": leakage_analysis,
        "silence_and_coverage": coverage_analysis,
    }

    if report_path:
        result["report_verification"] = compare_with_report(report_path, result)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze audio/video dubbing quality and alignment")
    parser.add_argument("--source", required=True, type=Path, help="Path to source video/audio")
    parser.add_argument("--dubbed", required=True, type=Path, help="Path to dubbed video/audio")
    parser.add_argument("--report", type=Path, default=None, help="Optional path to report.json")
    parser.add_argument("--segments", type=Path, default=None, help="Optional path to segments.<lang>.json")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write output JSON")

    args = parser.parse_args()

    results = analyze_dub(
        source_path=args.source,
        dubbed_path=args.dubbed,
        report_path=args.report,
        segments_path=args.segments,
    )

    formatted_json = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted_json, encoding="utf-8")
        print(f"Results saved to {args.output}")

    print("=" * 60)
    print("DUBBING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Container Match: {'PASS' if results['container']['pass'] else 'FAIL'} (Delta: {results['container']['duration_delta_ms']} ms)")
    print(f"Loudness Target: {'PASS' if results['loudness']['pass'] else 'FAIL'} (Dubbed: {results['loudness']['dubbed']['integrated_lufs']} LUFS, Peak: {results['loudness']['dubbed']['true_peak_dbtp']} dBTP)")
    print(f"Timing p95 Offset: {results['timing_alignment']['p95_abs_offset_ms']} ms (Max: {results['timing_alignment']['max_abs_offset_ms']} ms, >300ms count: {results['timing_alignment']['regions_beyond_300ms_count']})")
    print(f"Leading Gap: Source {results['timing_alignment']['leading_speech_gap_s']['source']}s vs Dubbed {results['timing_alignment']['leading_speech_gap_s']['dubbed']}s")
    print(f"Speech Mismatch: Dubbed speaks while source silent: {results['timing_alignment']['speech_mismatch_s']['dubbed_speech_source_silent_s']}s")
    print(f"Original Leakage: {results['original_audio_leakage']['total_high_correlation_windows']} windows (Mute bug: {results['original_audio_leakage']['mute_bug_detected']})")
    print(f"Dropped Speech: {results['silence_and_coverage']['dropped_speech_regions_count']} long silence drop regions")
    if "report_verification" in results:
        print(f"Report Claim Honest: {results['report_verification'].get('is_report_honest')}")
        if results['report_verification'].get('mismatches'):
            for m in results['report_verification']['mismatches']:
                print(f"  - Mismatch: {m['issue']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
