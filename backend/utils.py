import os

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
MAX_FILE_SIZE_MB = 200


def validate_video(file) -> str | None:
    """Returns error string if invalid, None if valid"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Invalid file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    return None


def generate_anomaly_report(results, threshold_low, threshold_high):
    """Generate a JSON-serializable summary report from detection results"""
    if not results:
        return {}

    total_frames = len(results)
    normal_frames = sum(1 for r in results if r['status'] == 'NORMAL')
    suspicious_frames = sum(1 for r in results if r['status'] == 'SUSPICIOUS')
    anomaly_frames = sum(1 for r in results if r['status'] == 'ANOMALY')

    scores = [r['anomaly_score'] for r in results]
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Overall threat level
    anomaly_ratio = anomaly_frames / total_frames if total_frames > 0 else 0
    suspicious_ratio = suspicious_frames / total_frames if total_frames > 0 else 0

    if anomaly_ratio > 0.3:
        threat_level = "HIGH"
    elif anomaly_ratio > 0.1 or suspicious_ratio > 0.3:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    # Find flagged segments (consecutive anomalous/suspicious frames)
    flagged_segments = []
    in_segment = False
    seg_start_pos = 0  # position in results list, not frame_idx

    for i, r in enumerate(results):
        if r['status'] in ('SUSPICIOUS', 'ANOMALY') and not in_segment:
            in_segment = True
            seg_start_pos = i
        elif r['status'] == 'NORMAL' and in_segment:
            in_segment = False
            segment_slice = results[seg_start_pos:i]
            if segment_slice:
                flagged_segments.append({
                    'start_frame': results[seg_start_pos]['frame_idx'],
                    'end_frame': results[i - 1]['frame_idx'],
                    'peak_score': max(r2['anomaly_score'] for r2 in segment_slice)
                })

    # Close any segment still open at the end of the video
    if in_segment:
        segment_slice = results[seg_start_pos:]
        if segment_slice:
            flagged_segments.append({
                'start_frame': results[seg_start_pos]['frame_idx'],
                'end_frame': results[-1]['frame_idx'],
                'peak_score': max(r['anomaly_score'] for r in segment_slice)
            })

    # Per-frame scores (lightweight — just index + score)
    frame_scores = [
        {
            'frame_idx': r['frame_idx'],
            'score': round(r['anomaly_score'], 6),
            'status': r['status']
        }
        for r in results
    ]

    return {
        'total_frames': total_frames,
        'normal_frames': normal_frames,
        'suspicious_frames': suspicious_frames,
        'anomaly_frames': anomaly_frames,
        'max_score': round(max_score, 6),
        'avg_score': round(avg_score, 6),
        'threat_level': threat_level,
        'threshold_low': round(threshold_low, 6),
        'threshold_high': round(threshold_high, 6),
        'flagged_segments': flagged_segments,
        'frame_scores': frame_scores
    }