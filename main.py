from pose_detection import detect_pose
from body_analysis import analyze_body
from body_classifier import classify_body
from formal_recommender import recommend_formal_style

import cv2
import numpy as np
import os


def resize_images(front, side, target_height=600):

    h1, w1 = front.shape[:2]
    h2, w2 = side.shape[:2]

    scale1 = target_height / h1
    scale2 = target_height / h2

    front = cv2.resize(front, (int(w1 * scale1), target_height))
    side = cv2.resize(side, (int(w2 * scale2), target_height))

    return front, side


# -----------------------------
# FRONT IMAGE ANALYSIS
# -----------------------------

front_image_path = os.path.join(os.path.dirname(__file__), "images", "front.jpg")
front_keypoints, front_image = detect_pose(front_image_path)

front_analysis = analyze_body(front_keypoints)

classification = classify_body(front_analysis)


# -----------------------------
# SIDE IMAGE
# -----------------------------

side_image_path = os.path.join(os.path.dirname(__file__), "images", "side.jpg")
side_keypoints, side_image = detect_pose(side_image_path)


# -----------------------------
# RESIZE IMAGES
# -----------------------------

front_image, side_image = resize_images(front_image, side_image)


# -----------------------------
# COMBINE FRONT + SIDE
# -----------------------------

combined = cv2.hconcat([front_image, side_image])


# -----------------------------
# GET RECOMMENDATIONS
# -----------------------------

recommendations = recommend_formal_style(classification)


print("\nSMARTFIT RESULT\n")

for r in recommendations:
    print("-", r)


# -----------------------------
# CREATE RECOMMENDATION PANEL
# -----------------------------

panel_width = 500
panel = np.zeros((combined.shape[0], panel_width, 3), dtype=np.uint8)

# Title
cv2.putText(panel,
            "SMARTFIT RESULT",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (240,240,255),
            2)

y = 120

for r in recommendations:

    cv2.putText(panel,
                "-> " + r,
                (20,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240,240,255),
                2)

    y += 50


# -----------------------------
# FINAL DISPLAY
# -----------------------------

final_view = cv2.hconcat([combined, panel])

cv2.imshow("SmartFit Analysis", final_view)

cv2.waitKey(0)
cv2.destroyAllWindows()