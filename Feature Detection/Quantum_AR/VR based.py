import cv2
import numpy as np

# Constants for AR
MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures=5000)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def load_input():
    input_image = cv2.imread('Horse.jpeg')
    augment_image = cv2.imread('The_kid_who_came_from_space_Camera.jpg')

    input_image = cv2.resize(input_image, (300, 400), interpolation=cv2.INTER_AREA)
    augment_image = cv2.resize(augment_image, (300, 400))
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    return gray_image, augment_image, keypoints, descriptors

def compute_matches(descriptors_input, descriptors_output):
    if len(descriptors_output) != 0 and len(descriptors_input) != 0:
        matches = flann.knnMatch(np.asarray(descriptors_input, np.float32), np.asarray(descriptors_output, np.float32), k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.69 * n.distance:
                good.append(m)
        return good
    else:
        return None

def create_stereo_pair(frame):
    height, width = frame.shape[:2]
    stereo_pair = np.zeros((height, width * 2, 3), dtype=np.uint8)
    stereo_pair[:, :width] = frame
    stereo_pair[:, width:] = frame
    return stereo_pair

def augment_frame(frame, input_image, augment_image, input_keypoints, input_descriptors):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_keypoints, frame_descriptors = detector.detectAndCompute(gray_frame, None)

    if frame_descriptors is None or len(frame_descriptors) == 0:
        return frame

    matches = compute_matches(input_descriptors, frame_descriptors)

    if matches is not None and len(matches) > MIN_MATCHES:
        src_pts = np.float32([input_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = input_image.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        warped_image = cv2.warpPerspective(augment_image, M, (frame.shape[1], frame.shape[0]))
        mask = np.zeros_like(frame)
        cv2.fillConvexPoly(mask, np.int32(dst), (255, 255, 255), cv2.LINE_AA)
        mask = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, warped_image)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    input_image, augment_image, input_keypoints, input_descriptors = load_input()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        augmented_frame = augment_frame(frame, input_image, augment_image, input_keypoints, input_descriptors)
        stereo_pair = create_stereo_pair(augmented_frame)

        cv2.imshow('AR and VR Augmentation', stereo_pair)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
