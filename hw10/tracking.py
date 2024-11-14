import cv2
import matplotlib.pyplot as plt

# Load video from the "data" folder and select object
video_path = "data/video.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")
cap.release()

tracker_csrt = cv2.TrackerCSRT_create()
tracker_kcf = cv2.TrackerKCF_create()
tracker_csrt.init(frame, roi)
tracker_kcf.init(frame, roi)


def track(video_path, tracker):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(20):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Update tracker
        success, box = tracker.update(frame)
        if success:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 4, 1)
        frames.append(frame)

    cap.release()

    # Display the frames using matplotlib
    for i, frame in enumerate(frames):
        plt.subplot(4, 5, i + 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {i + 1}")
        plt.axis("off")


plt.figure(figsize=(20, 20))
plt.suptitle("CSRT Tracker")
track(video_path, tracker_csrt)
plt.savefig("csrt_tracker.png")

plt.figure(figsize=(20, 20))
plt.suptitle("KCF Tracker")
track(video_path, tracker_kcf)
plt.savefig("kcf_tracker.png")

# not the most efficient way, but I create 2 copies to test other trackers
plt.show()
