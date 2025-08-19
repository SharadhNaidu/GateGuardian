import cv2
import face_recognition
import pickle
import numpy as np
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from config import *

class AdvancedGateController:
    def __init__(self):
        self.encodings_file = ENCODINGS_FILE
        self.known_face_encodings = {}
        self.face_metadata = {}
        self.gate_unlocked = False
        self.unlock_time = 0

        self.failed_attempts = defaultdict(int)
        self.lockout_until = defaultdict(float)
        self.recent_recognitions = deque(maxlen=FACE_MATCH_CONSENSUS)
        self.access_log = []

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        self.blink_detector = BlinkDetector() if ANTI_SPOOFING_ENABLED else None

        self.load_encodings()

    def load_encodings(self):
        """Load face encodings from file"""
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data.get('encodings', {})
                self.face_metadata = data.get('metadata', {})

            total_encodings = sum(len(encodings) for encodings in self.known_face_encodings.values())
            if total_encodings > 0:
                print(f"✅ Loaded {len(self.known_face_encodings)} people with {total_encodings} total encodings")
                return True
            else:
                print("❌ No face encodings found!")
                return False

        except FileNotFoundError:
            print("❌ No face encodings found! Please register faces first using add_face.py")
            return False
        except Exception as e:
            print(f"❌ Error loading encodings: {e}")
            return False

    def log_access_attempt(self, person_name, success, confidence=0, method="face_recognition"):
        """Log access attempts for security audit"""
        log_entry = {
            'timestamp': datetime.now(),
            'person': person_name,
            'success': success,
            'confidence': confidence,
            'method': method,
            'ip_address': 'local',
        }
        self.access_log.append(log_entry)

        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]

    def is_person_locked_out(self, person_name):
        """Check if person is temporarily locked out"""
        if person_name in self.lockout_until:
            if time.time() < self.lockout_until[person_name]:
                return True
            else:
                del self.lockout_until[person_name]
                self.failed_attempts[person_name] = 0
        return False

    def handle_failed_attempt(self, person_name):
        """Handle failed recognition attempts"""
        self.failed_attempts[person_name] += 1

        if self.failed_attempts[person_name] >= SECURITY_LOCKOUT_ATTEMPTS:
            self.lockout_until[person_name] = time.time() + LOCKOUT_DURATION
            print(f"🚫 {person_name} temporarily locked out for {LOCKOUT_DURATION} seconds")
            return True
        return False

    def recognize_person_ensemble(self, face_encoding):
        """Advanced face recognition using ensemble voting"""
        if not self.known_face_encodings:
            return "Unknown", 0.0

        all_encodings = []
        labels = []

        for name, encodings in self.known_face_encodings.items():
            for encoding in encodings:
                all_encodings.append(encoding)
                labels.append(name)

        if not all_encodings:
            return "Unknown", 0.0

        distances = face_recognition.face_distance(all_encodings, face_encoding)

        matches = distances <= RECOGNITION_THRESHOLD

        if not any(matches):
            return "Unknown", 0.0

        person_votes = defaultdict(list)
        for i, (distance, match) in enumerate(zip(distances, matches)):
            if match:
                person_name = labels[i]
                confidence = 1 - distance
                person_votes[person_name].append(confidence)

        person_scores = {}
        for person, confidences in person_votes.items():
            avg_confidence = np.mean(confidences)
            vote_bonus = min(len(confidences) / ENCODINGS_PER_PERSON, 1.0) * 0.1
            person_scores[person] = avg_confidence + vote_bonus

        best_person = max(person_scores, key=person_scores.get)
        best_confidence = person_scores[best_person]

        return best_person, best_confidence

    def unlock_gate(self, person_name, confidence):
        """Unlock the gate with security logging"""
        if self.is_person_locked_out(person_name):
            print(f"🚫 Access denied: {person_name} is temporarily locked out")
            self.log_access_attempt(person_name, False, confidence, "lockout")
            return False

        self.gate_unlocked = True
        self.unlock_time = time.time()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.failed_attempts[person_name] = 0

        self.log_access_attempt(person_name, True, confidence)

        print(f"\n🔓 GATE UNLOCKED for {person_name}")
        print(f"⏰ Time: {current_time}")
        print(f"📊 Confidence: {confidence:.3f}")
        print(f"⏱️  Gate will auto-lock in {GATE_UNLOCK_DURATION} seconds")


        return True

    def lock_gate(self):
        """Lock the gate"""
        self.gate_unlocked = False
        print("\n🔒 GATE LOCKED")

    def check_gate_timeout(self):
        """Check if gate should be locked due to timeout"""
        if self.gate_unlocked and (time.time() - self.unlock_time) > GATE_UNLOCK_DURATION:
            self.lock_gate()

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time

    def detect_image_quality(self, frame, face_location):
        """Detect if image quality is sufficient for recognition"""
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]

        if face_img.size == 0:
            return False, "Empty face region"

        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        brightness = np.mean(gray_face)

        is_good_quality = (blur_score > BLUR_DETECTION_THRESHOLD and
                          BRIGHTNESS_THRESHOLD[0] < brightness < BRIGHTNESS_THRESHOLD[1])

        quality_msg = f"Blur: {blur_score:.1f}, Brightness: {brightness:.1f}"

        return is_good_quality, quality_msg

    def recognize_face(self):
        """Main face recognition loop with advanced features"""
        if not self.known_face_encodings:
            print("❌ No registered faces found! Please run add_face.py first.")
            return

        print("🚀 Starting Advanced GateGuardian Face Recognition System...")
        print("📋 Loaded faces:", ", ".join(self.known_face_encodings.keys()))
        print("🎯 Press 'q' to quit, 's' for statistics")

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return

        process_this_frame = True
        frame_skip_count = 0
        recognition_results = []

        print("✅ Camera initialized. Starting recognition...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break

            self.update_fps()

            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    model=FACE_DETECTION_MODEL,
                    number_of_times_to_upsample=FACE_DETECTION_UPSAMPLING
                )

                recognition_results = []

                for face_location in face_locations:
                    top, right, bottom, left = [coord * 4 for coord in face_location]
                    scaled_location = (top, right, bottom, left)

                    is_good_quality, quality_msg = self.detect_image_quality(frame, scaled_location)

                    if is_good_quality:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_encodings = face_recognition.face_encodings(rgb_frame, [scaled_location])

                        if face_encodings:
                            name, confidence = self.recognize_person_ensemble(face_encodings[0])

                            if ANTI_SPOOFING_ENABLED and self.blink_detector:
                                blink_detected = self.blink_detector.detect_blink(frame, scaled_location)
                                if not blink_detected:
                                    name = "Spoofing?"
                                    confidence = 0.0

                            recognition_results.append({
                                'location': scaled_location,
                                'name': name,
                                'confidence': confidence,
                                'quality': quality_msg,
                                'is_good_quality': is_good_quality
                            })

                            if (name != "Unknown" and name != "Spoofing?" and
                                confidence > CONFIDENCE_VOTING_THRESHOLD and
                                not self.gate_unlocked):

                                self.recent_recognitions.append((name, confidence))

                                if len(self.recent_recognitions) >= FACE_MATCH_CONSENSUS:
                                    recent_names = [r[0] for r in self.recent_recognitions]
                                    if recent_names.count(name) >= FACE_MATCH_CONSENSUS:
                                        self.unlock_gate(name, confidence)
                                        self.recent_recognitions.clear()
                        else:
                            recognition_results.append({
                                'location': scaled_location,
                                'name': "No encoding",
                                'confidence': 0.0,
                                'quality': quality_msg,
                                'is_good_quality': is_good_quality
                            })
                    else:
                        recognition_results.append({
                            'location': scaled_location,
                            'name': "Poor quality",
                            'confidence': 0.0,
                            'quality': quality_msg,
                            'is_good_quality': is_good_quality
                        })

            process_this_frame = not process_this_frame

            for result in recognition_results:
                top, right, bottom, left = result['location']
                name = result['name']
                confidence = result['confidence']
                is_good_quality = result['is_good_quality']

                if name == "Unknown" or name == "No encoding":
                    color = COLOR_RED
                    label = name
                elif name == "Poor quality":
                    color = COLOR_ORANGE
                    label = "Poor Quality"
                elif name == "Spoofing?":
                    color = COLOR_YELLOW
                    label = "Spoofing Detected"
                elif confidence > CONFIDENCE_VOTING_THRESHOLD:
                    color = COLOR_GREEN
                    label = f"{name} ({confidence:.3f})"
                else:
                    color = COLOR_BLUE
                    label = f"{name}? ({confidence:.3f})"

                thickness = 3 if is_good_quality else 1
                cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

                label_height = 35 if SHOW_CONFIDENCE_SCORES else 25
                cv2.rectangle(frame, (left, bottom - label_height), (right, bottom), color, cv2.FILLED)

                font_scale = 0.6 if SHOW_CONFIDENCE_SCORES else 0.5
                cv2.putText(frame, label, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, font_scale, COLOR_WHITE, 1)

                if SHOW_CONFIDENCE_SCORES and result['quality']:
                    cv2.putText(frame, result['quality'], (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            self.draw_system_status(frame)

            self.check_gate_timeout()

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_statistics()

        cap.release()
        cv2.destroyAllWindows()

        if self.gate_unlocked:
            self.lock_gate()

    def draw_system_status(self, frame):
        """Draw system status information on frame"""
        status_color = COLOR_GREEN if self.gate_unlocked else COLOR_RED
        status_text = "🔓 UNLOCKED" if self.gate_unlocked else "🔒 LOCKED"
        cv2.putText(frame, f"Gate: {status_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

        cv2.putText(frame, f"Registered: {len(self.known_face_encodings)}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        if SHOW_FPS:
            cv2.putText(frame, f"FPS: {self.current_fps}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        active_lockouts = sum(1 for t in self.lockout_until.values() if t > time.time())
        if active_lockouts > 0:
            cv2.putText(frame, f"Lockouts: {active_lockouts}", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        if len(self.recent_recognitions) > 0:
            consensus_progress = len(self.recent_recognitions) / FACE_MATCH_CONSENSUS
            cv2.putText(frame, f"Consensus: {len(self.recent_recognitions)}/{FACE_MATCH_CONSENSUS}",
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

        cv2.putText(frame, "Press 'q' to quit, 's' for stats", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    def show_statistics(self):
        """Show detailed system statistics"""
        print("\n" + "="*60)
        print("📊 GateGuardian System Statistics")
        print("="*60)

        print(f"👥 Registered People: {len(self.known_face_encodings)}")
        for name, encodings in self.known_face_encodings.items():
            metadata = self.face_metadata.get(name, {})
            print(f"   • {name}: {len(encodings)} encodings (Quality: {metadata.get('avg_quality', 0):.2f})")

        print(f"\n🔒 Security Status:")
        print(f"   • Failed Attempts: {dict(self.failed_attempts)}")
        print(f"   • Active Lockouts: {sum(1 for t in self.lockout_until.values() if t > time.time())}")

        print(f"\n📋 Recent Access Log (Last 10):")
        for log in self.access_log[-10:]:
            status = "✅" if log['success'] else "❌"
            print(f"   {status} {log['timestamp'].strftime('%H:%M:%S')} - {log['person']} ({log['confidence']:.3f})")

        print(f"\n⚡ Performance:")
        print(f"   • Current FPS: {self.current_fps}")
        print(f"   • Total Access Attempts: {len(self.access_log)}")

        print("="*60)


class BlinkDetector:
    """Simple blink detection for anti-spoofing"""
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.blink_history = deque(maxlen=10)

    def detect_blink(self, frame, face_location):
        """Detect if person is blinking (basic anti-spoofing)"""
        top, right, bottom, left = face_location
        face_roi = frame[top:bottom, left:right]

        if face_roi.size == 0:
            return False

        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)

        eyes_detected = len(eyes) > 0
        self.blink_history.append(eyes_detected)

        if len(self.blink_history) >= 5:
            recent = list(self.blink_history)[-5:]
            blink_pattern = any(not recent[i] and recent[i-1] and recent[i+1] for i in range(1, 4))
            return blink_pattern

        return True


def main():
    print("🔐 GateGuardian - Advanced Face Recognition Gate Controller")
    print("=" * 60)

    controller = AdvancedGateController()

    if controller.known_face_encodings:
        print("✅ System ready!")
        controller.recognize_face()
    else:
        print("❌ No faces registered! Please run add_face.py first to register faces.")

if __name__ == "__main__":
    main()
