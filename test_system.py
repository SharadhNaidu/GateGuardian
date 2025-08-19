import cv2
import numpy as np
import time

def test_camera():
    """Test camera functionality and basic face detection"""
    print("🎥 Testing camera and basic face detection...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        print("Please check if your camera is connected and not being used by another application")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("❌ Error: Could not load face cascade classifier")
        return False

    print("✅ Camera initialized successfully!")
    print("📹 Camera resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("🎯 Press 'q' to quit, 's' to take screenshot")

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame from camera")
            break

        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {w}x{h}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"GateGuardian Camera Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {current_fps}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' for screenshot", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("GateGuardian Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_name = f"test_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"📸 Screenshot saved as: {screenshot_name}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed successfully!")
    return True

def test_opencv_features():
    """Test OpenCV advanced features"""
    print("\n🔬 Testing OpenCV advanced features...")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("✅ LBPH Face Recognizer available")
    except Exception as e:
        print(f"❌ LBPH Face Recognizer not available: {e}")

    try:
        eigenface = cv2.face.EigenFaceRecognizer_create()
        print("✅ EigenFace Recognizer available")
    except Exception as e:
        print(f"❌ EigenFace Recognizer not available: {e}")

    try:
        fisherface = cv2.face.FisherFaceRecognizer_create()
        print("✅ FisherFace Recognizer available")
    except Exception as e:
        print(f"❌ FisherFace Recognizer not available: {e}")

    return True

if __name__ == "__main__":
    print("🔐 GateGuardian System Test")
    print("=" * 50)

    test_opencv_features()

    if test_camera():
        print("\n✅ All tests passed! Your system is ready for GateGuardian.")
        print("\n📋 Next steps:")
        print("1. Run 'python add_face.py' to register faces")
        print("2. Run 'python unlock_gate.py' to start the gate controller")
    else:
        print("\n❌ Camera test failed. Please check your camera setup.")
