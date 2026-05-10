LABELS = [
    "agreement",
    "refusal",
    "mocking",
    "comfort",
    "confusion",
    "celebration",
    "neutral",
]

LABEL_DESCRIPTIONS = {
    "agreement": "agree, support, approve",
    "refusal": "reject, unwilling, no",
    "mocking": "sarcasm, teasing, ridicule",
    "comfort": "encouragement, care, consolation",
    "confusion": "confused, speechless, shocked",
    "celebration": "happy, excited, success",
    "neutral": "general meme",
}

KEYWORDS = {
    "agreement": [
        "可以",
        "赞",
        "支持",
        "同意",
        "没错",
        "确实",
        "好耶",
        "收到",
        "安排",
    ],
    "refusal": [
        "不要",
        "不行",
        "拒绝",
        "算了",
        "别",
        "不想",
        "达咩",
        "滚",
        "爬",
    ],
    "mocking": [
        "笑死",
        "哈哈",
        "呵呵",
        "就这",
        "离谱",
        "小丑",
        "绷不住",
        "乐",
    ],
    "comfort": [
        "抱抱",
        "加油",
        "没事",
        "安慰",
        "辛苦",
        "会好的",
        "摸摸",
        "别难过",
    ],
    "confusion": [
        "啊",
        "啥",
        "什么",
        "疑惑",
        "懵",
        "看不懂",
        "问号",
        "震惊",
        "无语",
    ],
    "celebration": [
        "开心",
        "快乐",
        "赢",
        "胜利",
        "庆祝",
        "起飞",
        "狂喜",
        "太棒",
        "恭喜",
    ],
}


def weak_label(text):
    if not isinstance(text, str):
        return "neutral"
    scores = {label: 0 for label in LABELS}
    for label, words in KEYWORDS.items():
        for word in words:
            if word in text:
                scores[label] += 1
    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        return "neutral"
    return best_label

