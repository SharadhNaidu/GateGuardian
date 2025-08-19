import cv2
import sys

def test_camera_simple():
    """Simple camera test to verify OpenCV window display"""
    print("🎥 Simple Camera Test")
    print("=" * 40)

    for camera_index in range(3):
        print(f"Testing camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {camera_index} is working!")
                print(f"📐 Frame size: {frame.shape}")

                window_name = f"Camera Test - Index {camera_index}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)

                print("🖼️  Showing camera feed for 10 seconds...")
                print("Press any key to continue or ESC to exit")

                frame_count = 0
                while frame_count < 300:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Cannot read frame")
                        break

                    cv2.putText(frame, f"Camera {camera_index} - Frame {frame_count}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press ESC to exit, any other key to continue",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.imshow(window_name, frame)

                    key = cv2.waitKey(33) & 0xFF
                    if key == 27:
                        print("ESC pressed - exiting")
                        break
                    elif key != 255:
                        print(f"Key pressed: {key}")
                        break

                    frame_count += 1

                cv2.destroyAllWindows()
                cap.release()

                if frame_count > 0:
                    print(f"✅ Successfully displayed {frame_count} frames")
                    return True
                else:
                    print("❌ Could not display frames")
            else:
                print(f"❌ Camera {camera_index} opened but cannot read frames")
        else:
            print(f"❌ Cannot open camera {camera_index}")

        cap.release()

    print("❌ No working cameras found")
    return False

def check_opencv_info():
    """Check OpenCV installation and capabilities"""
    print("\n🔍 OpenCV Information")
    print("=" * 40)
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Python Version: {sys.version}")

    backends = []
    if hasattr(cv2, 'CAP_DSHOW'):
        backends.append("DirectShow")
    if hasattr(cv2, 'CAP_MSMF'):
        backends.append("Media Foundation")
    if hasattr(cv2, 'CAP_V4L2'):
        backends.append("V4L2")

    print(f"Available backends: {', '.join(backends) if backends else 'Unknown'}")

    try:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        print("✅ GUI support: Available")
    except Exception as e:
        print(f"❌ GUI support: Not available - {e}")

if __name__ == "__main__":
    check_opencv_info()

    print("\nStarting camera test...")
    if test_camera_simple():
        print("\n✅ Camera test successful!")
        print("Your camera is working with OpenCV.")
        print("You can now run the face registration system.")
    else:
        print("\n❌ Camera test failed!")
        print("Possible solutions:")
        print("1. Check if camera is connected and working in other apps")
        print("2. Close any other applications using the camera")
        print("3. Run as administrator")
        print("4. Check Windows camera privacy settings")
        print("5. Try a different camera or USB port")
