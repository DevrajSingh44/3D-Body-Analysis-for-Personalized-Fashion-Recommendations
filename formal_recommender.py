def recommend_formal_style(classification):

    suggestions = []

    frame = classification["body_frame"]
    proportion = classification["proportion"]

    # Body frame styling
    if frame == "Broad Shoulders":
        suggestions.append("Use slim-fit blazers to balance shoulder width")
        suggestions.append("Avoid heavy shoulder padding")

    elif frame == "Hip Dominant":
        suggestions.append("Use structured jackets with defined shoulders")
        suggestions.append("Choose darker trousers")

    else:
        suggestions.append("Most formal styles will suit your balanced frame")

    # Proportion styling
    if proportion == "Long Torso":
        suggestions.append("Use mid or high-rise trousers")
        suggestions.append("Avoid very long jackets")

    elif proportion == "Long Legs":
        suggestions.append("Use longer blazers")
        suggestions.append("Mid-rise trousers maintain balance")

    else:
        suggestions.append("Standard blazer and trouser proportions work well")

    return suggestions

import cv2

def draw_body_proportions(image, keypoints):

    left_shoulder = keypoints.get(11)
    right_shoulder = keypoints.get(12)

    left_hip = keypoints.get(23)
    right_hip = keypoints.get(24)

    # Draw shoulder line
    cv2.line(image, left_shoulder, right_shoulder, (0,255,0), 3)

    # Draw hip line
    cv2.line(image, left_hip, right_hip, (255,0,0), 3)

    return image