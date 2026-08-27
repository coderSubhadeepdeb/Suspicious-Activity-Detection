import cv2
import numpy as np
import os


STATUS_COLORS = {
    "NORMAL":     (0, 200, 0),     # Green  (BGR)
    "SUSPICIOUS": (0, 200, 220),   # Yellow (BGR)
    "ANOMALY":    (0, 0, 220),     # Red    (BGR)
}


def _normalize_error_map(error_map, threshold_low, threshold_high):
    """
    Normalize the per-pixel reconstruction error against the ABSOLUTE
    threshold scale — not the local frame min/max.

    Old approach (broken):
        error_norm = (error - local_min) / (local_max - local_min)
        → Even a nearly-zero-error frame gets colours spanning 0→1,
          making heatmaps meaningless.

    New approach:
        0.0 corresponds to zero error (or below threshold_low).
        1.0 corresponds to threshold_high (full anomaly level).
        → Colours are consistent and meaningful across all frames.
    """
    # Use 3× threshold_high as the saturation point so that values at or
    # above the anomaly threshold are clearly saturated red, while values
    # well below threshold_low stay near zero.
    scale = max(threshold_high * 3.0, 1e-8)
    norm  = np.clip(error_map / scale, 0.0, 1.0)
    return norm.astype(np.float32)


def generate_heatmap_frame(original_frame, error_map, anomaly_score, status,
                            pixel_low, pixel_high,
                            all_scores=None, current_idx=None):
    """
    Overlay a meaningful heatmap on the original frame.

    Parameters
    ----------
    original_frame : np.ndarray   BGR frame from the video
    error_map      : np.ndarray   (H_model, W_model) per-pixel MSE from model
    anomaly_score  : float        smoothed anomaly score for this frame
    status         : str          'NORMAL' | 'SUSPICIOUS' | 'ANOMALY'
    pixel_low      : float        threshold_low from detect.py
    pixel_high     : float        threshold_high from detect.py
    all_scores     : list|None    all smoothed scores (for timeline bar)
    current_idx    : int|None     position in all_scores for the cursor
    """
    h_orig, w_orig = original_frame.shape[:2]

    # ── Resize & smooth error map ─────────────────────────────────────────────
    error_resized = cv2.resize(error_map, (w_orig, h_orig),
                               interpolation=cv2.INTER_CUBIC)
    error_resized = cv2.GaussianBlur(error_resized, (15, 15), 0)

    # ── Normalise against absolute threshold scale ────────────────────────────
    error_norm = _normalize_error_map(error_resized, pixel_low, pixel_high)

    # ── Build BGR heatmap using JET colormap ─────────────────────────────────
    # JET: blue (0) → cyan → green → yellow → red (1)
    error_uint8  = (error_norm * 255).astype(np.uint8)
    heatmap_jet  = cv2.applyColorMap(error_uint8, cv2.COLORMAP_JET)

    # Mask out very-low-error regions so they don't add visual clutter
    low_error_mask        = error_norm < 0.05
    heatmap_jet[low_error_mask] = [0, 0, 0]   # transparent-ish on blend

    # ── Blend with original frame ─────────────────────────────────────────────
    # Stronger heatmap for anomalous frames, lighter for normal ones
    heat_alpha = {
        "NORMAL":     0.15,
        "SUSPICIOUS": 0.30,
        "ANOMALY":    0.45,
    }.get(status, 0.25)
    overlay = cv2.addWeighted(original_frame, 1.0 - heat_alpha,
                              heatmap_jet,    heat_alpha, 0)

    # ── Status badge ─────────────────────────────────────────────────────────
    color = STATUS_COLORS.get(status, (200, 200, 200))
    cv2.rectangle(overlay, (10, 10), (320, 70), (20, 20, 20), -1)
    cv2.rectangle(overlay, (10, 10), (320, 70), color, 1)   # border
    cv2.putText(overlay, f"Status: {status}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(overlay, f"Score: {anomaly_score:.5f}  "
                         f"[L={pixel_low:.4f} H={pixel_high:.4f}]",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # ── Legend ────────────────────────────────────────────────────────────────
    lx = w_orig - 140
    cv2.rectangle(overlay, (lx, 10), (w_orig - 10, 115), (20, 20, 20), -1)
    cv2.putText(overlay, "Legend", (lx + 12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.circle(overlay, (lx + 18, 48),  7, (0, 160, 0),   -1)
    cv2.putText(overlay, "Normal",     (lx + 30, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.circle(overlay, (lx + 18, 68),  7, (0, 200, 220), -1)
    cv2.putText(overlay, "Suspicious", (lx + 30, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.circle(overlay, (lx + 18, 88),  7, (0, 0, 220),   -1)
    cv2.putText(overlay, "Anomaly",    (lx + 30, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    # ── Timeline bar (score graph at bottom) ─────────────────────────────────
    if all_scores is not None and current_idx is not None and len(all_scores) > 1:
        bar_h   = 40
        bar_y   = h_orig - bar_h
        bar_w   = w_orig

        # Dark background strip
        cv2.rectangle(overlay, (0, bar_y), (bar_w, h_orig), (15, 15, 15), -1)

        scores_arr = np.array(all_scores)
        # Normalise scores for display; clip at threshold_high * 2
        s_max = max(float(scores_arr.max()), pixel_high * 1.2, 1e-8)

        n = len(all_scores)
        for j in range(n - 1):
            x1 = int(j       / n * bar_w)
            x2 = int((j + 1) / n * bar_w)
            y1 = bar_y + bar_h - int(all_scores[j]     / s_max * (bar_h - 4))
            y2 = bar_y + bar_h - int(all_scores[j + 1] / s_max * (bar_h - 4))
            # Colour segment by local score
            seg_score = (all_scores[j] + all_scores[j + 1]) / 2
            if seg_score >= pixel_high:
                seg_col = (0, 0, 200)
            elif seg_score >= pixel_low:
                seg_col = (0, 200, 220)
            else:
                seg_col = (0, 180, 0)
            cv2.line(overlay, (x1, y1), (x2, y2), seg_col, 2)

        # Draw threshold lines
        tl_y = bar_y + bar_h - int(pixel_low  / s_max * (bar_h - 4))
        th_y = bar_y + bar_h - int(pixel_high / s_max * (bar_h - 4))
        cv2.line(overlay, (0, tl_y), (bar_w, tl_y), (0, 200, 220), 1)
        cv2.line(overlay, (0, th_y), (bar_w, th_y), (0, 0, 220),   1)

        # Current-frame cursor
        cx = int(current_idx / n * bar_w)
        cv2.line(overlay, (cx, bar_y), (cx, h_orig), (255, 255, 255), 1)

    return overlay


def generate_heatmap_video(results, threshold_low, threshold_high, output_path):
    if not results:
        print("No results to process.")
        return

    first_frame = results[0]['original_frame']
    h, w        = first_frame.shape[:2]

    # Pre-collect all overlay frames in memory
    all_scores = [r['anomaly_score'] for r in results]
    overlay_frames = []

    for idx, r in enumerate(results):
        status = r.get('status', 'NORMAL')
        frame_overlay = generate_heatmap_frame(
            r['original_frame'],
            r['error_map'],
            r['anomaly_score'],
            status,
            threshold_low,
            threshold_high,
            all_scores=all_scores,
            current_idx=idx
        )
        overlay_frames.append(frame_overlay)

    # 1. Try ImageIO first (reliable web-compatible H.264 MP4)
    try:
        import imageio
        # Ensure we write with libx264 and yuv420p for browser playability
        writer = imageio.get_writer(
            output_path, 
            fps=15.0, 
            codec='libx264', 
            pixelformat='yuv420p',
            ffmpeg_log_level='error'
        )
        for frame in overlay_frames:
            # imageio expects RGB, OpenCV is BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
        writer.close()
        print(f"Heatmap video successfully saved using ImageIO (H.264) to {output_path}")
        return
    except ImportError:
        print("ImageIO/FFmpeg not installed. Falling back to OpenCV...")

    # 2. Fallback to OpenCV VideoWriter (usually writes MPEG-4 'mp4v' on Windows)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 15.0, (w, h))
    if not out.isOpened():
        print("ERROR: OpenCV could not open video writer.")
        return

    for frame in overlay_frames:
        out.write(frame)
    out.release()
    print(f"Heatmap video saved using OpenCV (mp4v) to {output_path}")

    # 3. Try to transcode using local FFmpeg CLI if available
    import subprocess
    temp_path = output_path + ".temp.mp4"
    try:
        os.rename(output_path, temp_path)
        cmd = [
            "ffmpeg", "-y", "-i", temp_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        # Run ffmpeg silently to convert mp4v to H.264
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print("Heatmap video successfully transcoded to web-compatible H.264 using FFmpeg CLI.")
    except Exception:
        # Restore the original file if transcoding fails
        if os.path.exists(temp_path):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
        
        print("\n" + "="*80)
        print("WARNING: The generated MP4 file uses the 'mp4v' codec. This codec IS NOT")
        print("playable directly in web browsers (e.g. Chrome, Edge, Safari).")
        print("\nTo fix this so it plays on your website, run:")
        print("    uv pip install imageio imageio-ffmpeg")
        print("and process the video again.")
        print("="*80 + "\n")