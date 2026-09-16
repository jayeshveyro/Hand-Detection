# Hand Tracking 
It is a open-source Hand tracking project built to track multiple hand gestures.
## Hand Gestures
![](https://cdn.hackclub.com/01a0abd8-422c-78ee-94d2-0965177dd902/journal-1789589471077.png)
![](https://cdn.hackclub.com/01a0abd8-9c59-7ae5-933d-7ef8c7bf34aa/journal-1789589494149.png)
![](https://cdn.hackclub.com/01a0abd8-d41f-70ea-9c0e-f95abf712a59/journal-1789589508477.png)
![](https://cdn.hackclub.com/01a0abd8-fd6d-7e2d-8687-c69d1692d58e/journal-1789589519382.png)
![](https://cdn.hackclub.com/01a0abd9-6505-7f4a-8ed8-9ca7a5ab2155/journal-1789589545566.png)

## Code
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


def fingers_up(hand):
    """
    Returns:
    [thumb, index, middle, ring, pinky]

    1 = finger is up
    0 = finger is down
    """

    fingers = []

    # Thumb
    if hand[4].x > hand[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index
    if hand[8].y < hand[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle
    if hand[12].y < hand[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Ring
    if hand[16].y < hand[14].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Pinky
    if hand[20].y < hand[18].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers


def recognize_gesture(hand):
    fingers = fingers_up(hand)

    # Fist
    if fingers == [0, 0, 0, 0, 0]:
        return "FIST"

    # Open hand
    if fingers == [1, 1, 1, 1, 1]:
        return "OPEN HAND"

    # Pointing
    if fingers == [0, 1, 0, 0, 0]:
        return "POINTING"

    # Peace
    if fingers == [0, 1, 1, 0, 0]:
        return "PEACE"

    # Thumbs up
    if fingers == [1, 0, 0, 0, 0]:
        return "THUMBS UP"

    # I love you
    if fingers == [1, 1, 0, 0, 1]:
        return "I LOVE YOU"

    return "UNKNOWN"

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]

def draw_hand(frame, hand):
    height, width, _ = frame.shape


    for landmark in hand:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 255, 0),
            -1
        )

    for start, end in connections:

        x1 = int(hand[start].x * width)
        y1 = int(hand[start].y * height)

        x2 = int(hand[end].x * width)
        y2 = int(hand[end].y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

def main():

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hands
        result = detector.detect(mp_image)

        # Process hands
        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                draw_hand(frame, hand)

                gesture = recognize_gesture(hand)

                # Get wrist position
                height, width, _ = frame.shape

                x = int(hand[0].x * width)
                y = int(hand[0].y * height)

                # Display gesture
                cv2.putText(
                    frame,
                    gesture,
                    (x - 50, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

        cv2.imshow(
            "Hand Gesture Recognition",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
