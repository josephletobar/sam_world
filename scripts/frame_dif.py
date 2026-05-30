import cv2

def frame_dif(prev_frame, frame):

    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_curr, gray_prev)
    score = diff.mean()

    print(f"DIFFERENCE SCORE: {score}")

    if score > 50:
        run_gpt = True
    else:
        run_gpt = False

    return run_gpt