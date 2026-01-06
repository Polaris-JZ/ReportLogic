from collections import defaultdict
def keywords_tensorboards(tensorboard, responses, tensorboard_step):
    reflection_words = [
        "wait",
        "alternatively",
        "try",
        "other",
        "another",
        'hold on',
        'wait a moment',
        'double-',
        "recompute",
        'recheck',
        'we need to check',
        're-evaluate',
        'verify',
        "</think>",
        '<think>'
    ]
    rw_cnt = defaultdict(int)
    for response in responses:
        for rw_idx, rw in enumerate(reflection_words):
            if rw in response.lower():
                rw_cnt[rw] += 1
    for rw_idx, rw in enumerate(reflection_words):
        tensorboard.add_scalar(f"keywords/{rw}", rw_cnt[rw], tensorboard_step) 