"""
Quick test to demonstrate the optimized face detection system
"""

import cv2
import time
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

def quick_camera_test():
    """Quick test to show camera responsiveness"""
    print("🚀 GateGuardian - Quick Performance Test")
    print("=" * 50)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Camera not found")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print(f"✅ Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print("📹 Press 'q' to quit, 's' to take screenshot")
    print("⚡ Performance optimizations: HOG model, reduced resolution")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"📊 Frame {frame_count}: {len(faces)} face(s), FPS: {fps:.1f}")

        cv2.putText(frame, f"Frames: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if frame_count > 0:
            fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('GateGuardian - Performance Test', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"performance_test_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"📸 Screenshot saved: {screenshot_name}")

    cap.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f"\n📈 Performance Summary:")
    print(f"   Total frames: {frame_count}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average FPS: {avg_fps:.2f}")
    print(f"   Status: {'✅ EXCELLENT' if avg_fps > 20 else '⚠️ OK' if avg_fps > 10 else '❌ NEEDS OPTIMIZATION'}")

    return True

if __name__ == "__main__":
    quick_camera_test()
