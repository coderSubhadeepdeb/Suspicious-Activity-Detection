import cv2
import numpy as np
import os


STATUS_COLORS = {
    "NORMAL":     (0, 200, 0),     # Green (BGR)
    "SUSPICIOUS": (0, 200, 220),   # Yellow
    "ANOMALY":    (0, 0, 220),     # Red
}


def generate_heatmap_frame(original_frame, error_map, anomaly_score, status,
                            pixel_low, pixel_high):
    h_orig, w_orig = original_frame.shape[:2]

    # Resize error map to original frame size (smooth, not blocky)
    error_resized = cv2.resize(error_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    error_resized = cv2.GaussianBlur(error_resized, (15, 15), 0)

    # Normalize per-pixel error relative to THIS frame's own pixel range,
    # not a fixed global threshold — avoids the "everything turns red" issue
    local_min = error_resized.min()
    local_max = error_resized.max()
    if local_max - local_min < 1e-8:
        error_norm = np.zeros_like(error_resized)
    else:
        error_norm = (error_resized - local_min) / (local_max - local_min)

    # Only show heat where error is meaningfully above the frame's own baseline
    # (top 30% of this frame's own error distribution)
    display_thresh = np.percentile(error_norm, 70)
    error_norm = np.clip((error_norm - display_thresh) / (1 - display_thresh + 1e-8), 0, 1)

    heatmap_color = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)

    green_mask  = error_norm < 0.33
    yellow_mask = (error_norm >= 0.33) & (error_norm < 0.66)
    red_mask    = error_norm >= 0.66

    heatmap_color[green_mask] = [0, 160, 0]

    if yellow_mask.any():
        t = (error_norm[yellow_mask] - 0.33) / 0.33
        g_vals = (200 - 60 * t).astype(np.uint8)
        heatmap_color[yellow_mask] = np.stack(
            [np.zeros_like(g_vals), g_vals, np.full_like(g_vals, 220)], axis=1
        )

    heatmap_color[red_mask] = [0, 0, 220]

    # Blend heatmap with original frame — lighter blend so original content stays visible
    overlay = cv2.addWeighted(original_frame, 0.75, heatmap_color, 0.25, 0)

    # Status badge uses the REAL status computed in detect.py (passed in directly)
    color = STATUS_COLORS.get(status, (200, 200, 200))

    cv2.rectangle(overlay, (10, 10), (300, 60), (30, 30, 30), -1)
    cv2.putText(overlay, f"Status: {status}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(overlay, f"Score: {anomaly_score:.5f}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Legend
    legend_x = w_orig - 130
    cv2.rectangle(overlay, (legend_x, 10), (w_orig - 10, 100), (30, 30, 30), -1)
    cv2.putText(overlay, "Legend", (legend_x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(overlay, (legend_x + 20, 45), 7, (0, 160, 0), -1)
    cv2.putText(overlay, "Normal", (legend_x + 32, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(overlay, (legend_x + 20, 65), 7, (0, 200, 220), -1)
    cv2.putText(overlay, "Suspicious", (legend_x + 32, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(overlay, (legend_x + 20, 85), 7, (0, 0, 220), -1)
    cv2.putText(overlay, "Anomaly", (legend_x + 32, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return overlay


def generate_heatmap_video(results, threshold_low, threshold_high, output_path):
    if not results:
        print("No results to process.")
        return

    first_frame = results[0]['original_frame']
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, 15.0, (w, h))

    for r in results:
        # Use the ACTUAL status computed by detect.py's adaptive logic,
        # not a re-derivation from fixed thresholds
        status = r.get('status', 'NORMAL')

        frame_overlay = generate_heatmap_frame(
            r['original_frame'],
            r['error_map'],
            r['anomaly_score'],
            status,
            threshold_low,
            threshold_high
        )
        out.write(frame_overlay)

    out.release()
    print(f"Heatmap video saved to {output_path}")