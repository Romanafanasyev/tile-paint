from __future__ import annotations

import os
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _get_duration_seconds(path: str) -> float:
    rdr = imageio.get_reader(path)
    try:
        meta = rdr.get_meta_data()
    finally:
        rdr.close()
    dur = meta.get("duration")
    if dur and dur > 0:
        return float(dur)
    fps = meta.get("fps")
    nframes = meta.get("nframes")
    if fps and nframes and fps > 0:
        return float(nframes) / float(fps)
    raise RuntimeError("Cannot determine video duration")


def _has_audio_track(path: str, ffmpeg_bin: str) -> bool:
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
    )
    return "Audio:" in (proc.stderr or "")


def _clamp_segments(
    total: float, slow_sec: float, fast_sec: float, margin: float = 0.02
) -> tuple[float, float]:
    slow_sec = max(0.0, slow_sec)
    fast_sec = max(0.0, fast_sec)
    if slow_sec + fast_sec <= max(0.0, total - margin):
        return slow_sec, fast_sec
    if total <= margin:
        return 0.0, 0.0
    scale = (total - margin) / max(1e-9, (slow_sec + fast_sec))
    return slow_sec * scale, fast_sec * scale


def _atempo_chain_str(speed: float) -> str:
    chain: list[str] = []
    s = float(speed)
    if abs(s - 1.0) < 1e-6:
        return ""
    while s > 2.0 + 1e-9:
        chain.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        chain.append("atempo=0.5")
        s /= 0.5
    residual = f"{s:.6f}".rstrip("0").rstrip(".")
    chain.append(f"atempo={residual}")
    return ",".join(chain)


def _build_filter_complex(
    duration: float,
    slow_sec: float,
    fast_sec: float,
    steps: int,
    with_audio: bool,
    slow_from: float,
    fast_to: float,
) -> tuple[str, str, str | None]:
    vin, ain = "0:v", "0:a"
    parts_v, parts_a, lines = [], [], []

    def add_segment(t1: float, t2: float, speed: float, tag: str) -> None:
        v_label = f"[v{tag}]"
        lines.append(
            f"[{vin}]trim=start={t1:.6f}:end={t2:.6f},setpts=PTS-STARTPTS,setpts=PTS/{speed:.6f}{v_label}"
        )
        parts_v.append(v_label)
        if with_audio:
            a_label = f"[a{tag}]"
            chain = _atempo_chain_str(speed)
            if chain:
                lines.append(
                    f"[{ain}]atrim=start={t1:.6f}:end={t2:.6f},asetpts=PTS-STARTPTS,{chain}{a_label}"
                )
            else:
                lines.append(
                    f"[{ain}]atrim=start={t1:.6f}:end={t2:.6f},asetpts=PTS-STARTPTS{a_label}"
                )
            parts_a.append(a_label)

    if slow_sec > 1e-6:
        dt = slow_sec / steps
        for i in range(steps):
            t1 = i * dt
            t2 = (i + 1) * dt
            u = i / (steps - 1) if steps > 1 else 1.0
            s = slow_from + (1.0 - slow_from) * _smoothstep(u)
            add_segment(t1, t2, s, f"s{i}")

    mid_start, mid_end = slow_sec, max(slow_sec, duration - fast_sec)
    if mid_end - mid_start > 1e-6:
        v_mid = "[vMid]"
        lines.append(
            f"[{vin}]trim=start={mid_start:.6f}:end={mid_end:.6f},setpts=PTS-STARTPTS{v_mid}"
        )
        parts_v.append(v_mid)
        if with_audio:
            a_mid = "[aMid]"
            lines.append(
                f"[{ain}]atrim=start={mid_start:.6f}:end={mid_end:.6f},asetpts=PTS-STARTPTS{a_mid}"
            )
            parts_a.append(a_mid)

    if fast_sec > 1e-6:
        dt = fast_sec / steps
        base = duration - fast_sec
        for i in range(steps):
            t1 = base + i * dt
            t2 = base + (i + 1) * dt
            u = i / (steps - 1) if steps > 1 else 1.0
            s = 1.0 + (fast_to - 1.0) * _smoothstep(u)
            add_segment(t1, t2, s, f"f{i}")

    n = len(parts_v)
    if n == 0:
        raise RuntimeError("Empty segment set for concat")
    if with_audio:
        if len(parts_v) != len(parts_a):
            raise RuntimeError("Video/audio segments count mismatch")
        interleaved = []
        for v, a in zip(parts_v, parts_a):
            interleaved.extend([v, a])
        lines.append("".join(interleaved) + f"concat=n={n}:v=1:a=1[vout][aout]")
        return ";".join(lines), "[vout]", "[aout]"
    else:
        lines.append("".join(parts_v) + f"concat=n={n}:v=1:a=0[vout]")
        return ";".join(lines), "[vout]", None


def apply_speed_ramp_inplace(
    input_path: str,
    slow_seconds: float,
    fast_seconds: float,
    steps: int,
    crf: int,
    preset: str,
    fps: float | None,
    slow_from: float,
    fast_to: float,
) -> str:
    """
    Apply smooth speed ramp and replace the input file (temp file + atomic rename).
    """
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    duration = _get_duration_seconds(input_path)
    slow_sec, fast_sec = _clamp_segments(duration, slow_seconds, fast_seconds)
    with_audio = _has_audio_track(input_path, ffmpeg_bin)

    filter_complex, vout, aout = _build_filter_complex(
        duration=duration,
        slow_sec=slow_sec,
        fast_sec=fast_sec,
        steps=steps,
        with_audio=with_audio,
        slow_from=slow_from,
        fast_to=fast_to,
    )

    base, ext = os.path.splitext(input_path)
    tmp_out = f"{base}.speedtmp{ext}"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-filter_complex",
        filter_complex,
        "-map",
        vout,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if fps:
        cmd += ["-r", str(fps)]
    if aout:
        cmd += ["-map", aout, "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", tmp_out]

    subprocess.run(cmd, check=True)

    os.replace(tmp_out, input_path)
    return input_path
