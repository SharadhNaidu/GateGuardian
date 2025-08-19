import cv2
import face_recognition
import os
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time
from config import *

class AdvancedFaceRegistration:
    def __init__(self):
        self.faces_dir = FACES_DIR
        self.encodings_file = ENCODINGS_FILE
        self.training_data_file = TRAINING_DATA_FILE
        self.known_face_encodings = {}
        self.face_metadata = {}

        self.init_face_detection()

        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

        self.load_encodings()

    def init_face_detection(self):
        """Initialize face detection models"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        print("Advanced face detection models initialized")

    def calculate_image_quality(self, image):
        """Calculate image quality metrics"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        brightness = np.mean(gray)

        contrast = gray.std()

        return {
            'blur': blur_score,
            'brightness': brightness,
            'contrast': contrast,
            'quality_score': min(blur_score / 100, 1.0) * min(contrast / 50, 1.0)
        }

    def cleanup_windows(self):
        """Force cleanup all OpenCV windows"""
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process any pending window events
            time.sleep(0.1)
        except:
            pass

    def detect_face_landmarks(self, image):
        """Detect face landmarks for quality assessment"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(rgb_image)
        return len(face_landmarks_list) > 0

    def show_instruction_screen(self, instruction, duration=3, countdown=True):
        """Show a full-screen instruction with countdown"""
        print(f"Instruction: {instruction}")
        
        if countdown:
            for i in range(duration, 0, -1):
                # Create a black screen for instructions with proper size
                instruction_screen = np.zeros((750, 1000, 3), dtype=np.uint8)  # Match window size
                
                # Add instruction text
                lines = instruction.split('\n') if '\n' in instruction else [instruction]
                y_start = 300
                for j, line in enumerate(lines):
                    if line.strip():  # Only process non-empty lines
                        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                        x = (1000 - text_size[0]) // 2
                        y = y_start + j * 60
                        cv2.putText(instruction_screen, line, (x, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
                
                # Add countdown
                countdown_text = f"Starting in {i} seconds..."
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                x = (1000 - text_size[0]) // 2
                cv2.putText(instruction_screen, countdown_text, (x, 550), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_YELLOW, 3)
                
                # Add progress bar
                bar_width = 600
                bar_height = 30
                bar_x = (1000 - bar_width) // 2
                bar_y = 600
                
                progress = (duration - i) / duration
                fill_width = int(bar_width * progress)
                
                cv2.rectangle(instruction_screen, (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height), COLOR_WHITE, 2)
                if fill_width > 0:
                    cv2.rectangle(instruction_screen, (bar_x + 2, bar_y + 2), 
                                 (bar_x + fill_width - 2, bar_y + bar_height - 2), COLOR_GREEN, -1)
                
                cv2.imshow(WINDOW_NAME, instruction_screen)
                key = cv2.waitKey(1000)  # Wait 1 second
                if key == ord('q'):  # Allow early exit
                    break
        else:
            time.sleep(duration)

    def align_face(self, image, face_location):
        """Align face for better encoding"""
        if not ENABLE_FACE_ALIGNMENT:
            return image

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = face_recognition.face_landmarks(rgb_image, [face_location])

            if landmarks:
                left_eye = landmarks[0].get('left_eye', [])
                right_eye = landmarks[0].get('right_eye', [])

                if left_eye and right_eye:
                    # Convert to numpy arrays for proper calculation
                    left_eye = np.array(left_eye, dtype=np.float32)
                    right_eye = np.array(right_eye, dtype=np.float32)
                    
                    left_eye_center = np.mean(left_eye, axis=0)
                    right_eye_center = np.mean(right_eye, axis=0)

                    dy = right_eye_center[1] - left_eye_center[1]
                    dx = right_eye_center[0] - left_eye_center[0]
                    angle = np.degrees(np.arctan2(dy, dx))

                    # Properly convert center to tuple of integers
                    center_point = np.mean([left_eye_center, right_eye_center], axis=0)
                    center = (int(center_point[0]), int(center_point[1]))
                    
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                    return aligned
        except Exception as e:
            print(f"Face alignment failed: {e}")

        return image

    def is_quality_acceptable(self, quality_metrics):
        """Check if image quality is acceptable"""
        if not ENABLE_QUALITY_FILTERING:
            return True

        return (quality_metrics['blur'] > BLUR_DETECTION_THRESHOLD and
                BRIGHTNESS_THRESHOLD[0] < quality_metrics['brightness'] < BRIGHTNESS_THRESHOLD[1] and
                quality_metrics['contrast'] > 20)

    def load_encodings(self):
        """Load existing face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', {})
                    self.face_metadata = data.get('metadata', {})

                total_encodings = sum(len(encodings) for encodings in self.known_face_encodings.values())
                print(f"Loaded {len(self.known_face_encodings)} people with {total_encodings} total encodings")
            except Exception as e:
                print(f"Error loading encodings: {e}")
                self.known_face_encodings = {}
                self.face_metadata = {}

    def save_encodings(self):
        """Save face encodings to file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'metadata': self.face_metadata,
                'version': '2.0',
                'timestamp': time.time()
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print("Advanced face encodings saved successfully!")
        except Exception as e:
            print(f"Error saving encodings: {e}")

    def capture_multiple_encodings(self, name):
        """Capture multiple high-quality face encodings for a person"""
        if not name or name.strip() == "":
            print("Please provide a valid name!")
            return False

        if name in self.known_face_encodings:
            print(f"Face with name '{name}' already exists!")
            return False

        print(f"Starting face registration for: {name}")
        print("Follow the simple instructions for best results")

        # Show initial instruction screen
        self.show_instruction_screen(
            "Face Registration\n\n" +
            "Position yourself in front of the camera\n" +
            "Follow the pose instructions", 
            duration=3
        )

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            print("Trying alternative camera indices...")
            for i in range(3):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"✅ Found camera at index {i}")
                    break
                cap.release()
            else:
                print("❌ No camera found. Please check your camera connection.")
                return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        ret, test_frame = cap.read()
        if not ret:
            print("❌ Error: Camera opened but cannot read frames")
            cap.release()
            return False

        print("Camera initialized successfully!")
        print(f"Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1000, 750)  # Increased window size
        cv2.moveWindow(WINDOW_NAME, 50, 50)  # Better positioning

        cv2.imshow(WINDOW_NAME, test_frame)
        cv2.waitKey(1)

        print("Camera preview is ready!")

        captured_encodings = []
        captured_images = []
        quality_scores = []

        pose_guidance = [
            ("Look straight ahead", "Keep your head straight and look at the camera"),
            ("Turn right slightly", "Turn your head about 15 degrees to your right"),
            ("Turn left slightly", "Turn your head about 15 degrees to your left"),
            ("Tilt up slightly", "Gently tilt your chin up by about 10 degrees"),
            ("Tilt down slightly", "Gently tilt your chin down by about 10 degrees")
        ]
        current_pose = 0
        pose_instruction_time = time.time()
        pose_hold_duration = 4.0  # Increased duration for better guidance

        auto_start_timer = 0
        capture_ready = False
        quality_scores = []
        target_samples = ENCODINGS_PER_PERSON

        print("Starting face detection...")
        
        # Show positioning instruction
        self.show_instruction_screen(
            "Face Detection\n\n" +
            "Position your face in the frame\n" +
            "Good lighting is important", 
            duration=2
        )
        
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read from camera")
                break

            frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            opencv_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

            face_locations = []
            for (x, y, w, h) in opencv_faces:
                top, right, bottom, left = y, x + w, y + h, x
                face_locations.append((top, right, bottom, left))

            if len(face_locations) == 1:
                auto_start_timer += 1
                countdown = max(0, 90 - auto_start_timer)
                if countdown > 0:
                    cv2.putText(frame, f"Starting in {countdown//30 + 1} seconds...",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)
                    cv2.putText(frame, "Get ready!", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
                else:
                    print("Starting guided capture...")
                    # Show transition screen
                    self.show_instruction_screen(
                        "Face Detected!\n\n" +
                        "Starting guided capture\n" +
                        "Follow the instructions", 
                        duration=2
                    )
                    break
            else:
                auto_start_timer = 0
                if len(face_locations) == 0:
                    cv2.putText(frame, "No face detected - Position yourself",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
                else:
                    cv2.putText(frame, "Multiple faces - Only one person please",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 2)

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), COLOR_GREEN, 2)

                face_width = right - left
                face_height = bottom - top
                cv2.putText(frame, f"Face: {face_width}x{face_height}", (left, top - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1)

                face_img = frame[top:bottom, left:right]
                if face_img.size > 0:
                    quality = self.calculate_image_quality(face_img)
                    cv2.putText(frame, f"Quality: {quality['quality_score']:.2f}", (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)

            cv2.putText(frame, f"Face Registration: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BLUE, 2)
            cv2.putText(frame, f"Samples: {target_samples}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

            cv2.imshow(WINDOW_NAME, frame)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

            key = cv2.waitKey(30) & 0xFF

            if frame_count % 30 == 0:
                print(f"📹 Frame {frame_count}: {len(face_locations)} face(s) detected")

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False

        print("Starting guided capture...")
        
        # Show the first pose instruction
        current_instruction, detailed_instruction = pose_guidance[0]
        self.show_instruction_screen(
            f"Pose 1: {current_instruction}\n\n" +
            detailed_instruction, 
            duration=2
        )
        
        capture_start_time = time.time()
        frames_since_last_capture = 0
        min_frames_between_captures = 45  # Reduced for better responsiveness
        last_capture_flash = 0

        current_pose = 0
        pose_change_timer = 0
        pose_displayed_time = time.time()
        pose_transition_shown = False

        while len(captured_encodings) < target_samples:
            ret, frame = cap.read()
            if not ret:
                break

            frames_since_last_capture += 1
            pose_change_timer += 1

            time_on_current_pose = time.time() - pose_displayed_time
            if current_pose < len(pose_guidance) - 1 and time_on_current_pose > pose_hold_duration:
                current_pose += 1
                pose_displayed_time = time.time()
                pose_change_timer = 0
                pose_transition_shown = False
                
                # Show transition instruction screen
                current_instruction, detailed_instruction = pose_guidance[current_pose]
                print(f"📍 Moving to pose {current_pose + 1}: {current_instruction}")
                
                # Show instruction screen with delay
                self.show_instruction_screen(
                    f"Pose {current_pose + 1}: {current_instruction}\n\n" +
                    detailed_instruction, 
                    duration=2
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            opencv_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))

            face_locations = []
            for (x, y, w, h) in opencv_faces:
                top, right, bottom, left = y, x + w, y + h, x
                face_locations.append((top, right, bottom, left))

            image_captured_this_frame = False

            if len(face_locations) == 1 and frames_since_last_capture >= min_frames_between_captures:
                top, right, bottom, left = face_locations[0]

                face_img = frame[top:bottom, left:right]

                if face_img.size > 0:
                    quality = self.calculate_image_quality(face_img)

                    if self.is_quality_acceptable(quality):
                        # Use the original frame for better quality, not aligned frame for now
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        fr_face_locations = face_recognition.face_locations(rgb_image, model=FACE_DETECTION_MODEL, number_of_times_to_upsample=FACE_DETECTION_UPSAMPLING)
                        if fr_face_locations:
                            encodings = face_recognition.face_encodings(rgb_image, fr_face_locations)
                        else:
                            encodings = []

                        if encodings:
                            is_unique = True
                            if captured_encodings:
                                distances = face_recognition.face_distance(captured_encodings, encodings[0])
                                if min(distances) < 0.3:  # Ensure variety in poses
                                    is_unique = False

                            if is_unique:
                                captured_encodings.append(encodings[0])
                                # Save the properly cropped face region
                                face_region = frame[top:bottom, left:right]
                                captured_images.append(face_region.copy())
                                quality_scores.append(quality)
                                frames_since_last_capture = 0
                                last_capture_flash = time.time()
                                image_captured_this_frame = True

                                print(f"📸 CAPTURED sample {len(captured_encodings)}/{target_samples} (Quality: {quality['quality_score']:.2f})")

            flash_overlay = None
            if time.time() - last_capture_flash < 0.3:
                flash_alpha = max(0, 1 - (time.time() - last_capture_flash) / 0.3)
                flash_overlay = np.ones_like(frame) * 255
                flash_overlay = (flash_overlay * flash_alpha).astype(np.uint8)

            for i, (top, right, bottom, left) in enumerate(face_locations):
                if frames_since_last_capture >= min_frames_between_captures:
                    face_img = frame[top:bottom, left:right]
                    if face_img.size > 0:
                        quality = self.calculate_image_quality(face_img)
                        if self.is_quality_acceptable(quality):
                            color = COLOR_GREEN
                            status_text = "READY"
                        else:
                            color = COLOR_YELLOW
                            status_text = f"QUALITY: {quality['quality_score']:.2f}"
                    else:
                        color = COLOR_RED
                        status_text = "TOO SMALL"
                else:
                    color = COLOR_ORANGE
                    status_text = f"WAIT {(min_frames_between_captures - frames_since_last_capture)//30 + 1}s"

                if image_captured_this_frame:
                    color = COLOR_WHITE
                    status_text = "CAPTURED!"

                # Draw face rectangle with thicker border for better visibility
                cv2.rectangle(frame, (left, top), (right, bottom), color, 5)

                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = max(0, left + (right - left - text_size[0]) // 2)
                text_y = max(20, top - 15)

                # Add background for text for better readability
                cv2.rectangle(frame, (text_x - 5, text_y - 25), (text_x + text_size[0] + 5, text_y + 5), color, -1)
                cv2.putText(frame, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLACK, 2)

                if face_img.size > 0:
                    quality = self.calculate_image_quality(face_img)
                    # Quality indicators with better positioning
                    blur_color = COLOR_GREEN if quality['blur'] > BLUR_DETECTION_THRESHOLD else COLOR_RED
                    cv2.circle(frame, (left + 15, top + 40), 8, blur_color, -1)
                    cv2.putText(frame, "BLUR", (left + 30, top + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

                    brightness_ok = BRIGHTNESS_THRESHOLD[0] < quality['brightness'] < BRIGHTNESS_THRESHOLD[1]
                    bright_color = COLOR_GREEN if brightness_ok else COLOR_RED
                    cv2.circle(frame, (left + 15, top + 60), 8, bright_color, -1)
                    cv2.putText(frame, "LIGHT", (left + 30, top + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

            if flash_overlay is not None:
                frame = cv2.addWeighted(frame, 0.7, flash_overlay, 0.3, 0)

            progress = len(captured_encodings) / target_samples
            bar_width = 400
            bar_height = 30
            bar_x, bar_y = 10, 130

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), COLOR_BLACK, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), COLOR_WHITE, 2)

            fill_width = int(bar_width * progress)
            if fill_width > 0:
                cv2.rectangle(frame, (bar_x + 2, bar_y + 2), (bar_x + fill_width - 2, bar_y + bar_height - 2), COLOR_GREEN, -1)

            progress_text = f"Captured: {len(captured_encodings)}/{target_samples} ({progress*100:.1f}%)"
            text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = bar_x + (bar_width - text_size[0]) // 2
            cv2.putText(frame, progress_text, (text_x, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

            cv2.putText(frame, f"Face Registration: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BLUE, 2)

            # Show current pose instruction
            current_instruction, _ = pose_guidance[min(current_pose, len(pose_guidance)-1)]
            cv2.putText(frame, f"{current_instruction}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)

            # Show pose progress
            pose_progress_text = f"Pose {current_pose + 1}/{len(pose_guidance)}"
            cv2.putText(frame, pose_progress_text, (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

            time_remaining = max(0, pose_hold_duration - (time.time() - pose_displayed_time))
            if time_remaining > 0 and current_pose < len(pose_guidance) - 1:
                cv2.putText(frame, f"Hold for {time_remaining:.1f}s", (10, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
            elif current_pose >= len(pose_guidance) - 1:
                cv2.putText(frame, "Final pose", (10, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)

            cv2.putText(frame, "Press 'q' to quit", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

            if len(face_locations) == 1 and frames_since_last_capture < min_frames_between_captures:
                countdown_seconds = (min_frames_between_captures - frames_since_last_capture) / 30.0
                countdown_text = f"Next capture in: {countdown_seconds:.1f}s"
                cv2.putText(frame, countdown_text, (10, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

            # Simple status messages
            if len(face_locations) == 0:
                cv2.putText(frame, "No face detected", 
                           (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
            elif len(face_locations) > 1:
                cv2.putText(frame, "Multiple faces detected",
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if time.time() - capture_start_time > 60:
                print("⏰ Capture timeout. Using captured samples.")
                break

        cap.release()
        self.cleanup_windows()

        # Wait a moment for cleanup
        time.sleep(0.5)

        if len(captured_encodings) >= 3:
            print("\n" + "="*40)
            print("REGISTRATION COMPLETE!")
            print("="*40)
            print(f"Captured {len(captured_encodings)} samples")
            print("Processing and saving...")
            
            try:
                self.known_face_encodings[name] = captured_encodings
                self.face_metadata[name] = {
                    'quality_scores': quality_scores,
                    'capture_time': time.time(),
                    'num_samples': len(captured_encodings),
                    'avg_quality': np.mean([q['quality_score'] for q in quality_scores])
                }

                best_idx = np.argmax([q['quality_score'] for q in quality_scores])
                face_image_path = os.path.join(self.faces_dir, f"{name}.jpg")
                cv2.imwrite(face_image_path, captured_images[best_idx])

                self.save_encodings()

                print(f"Face registered successfully for {name}!")
                print(f"Quality score: {self.face_metadata[name]['avg_quality']:.2f}")
                print("="*40)
                print("Returning to main menu...")

                return True

            except Exception as e:
                print(f"Error processing face: {e}")
                return False
        else:
            print(f"Not enough samples captured ({len(captured_encodings)}/5)")
            print("Please try again with better lighting")
            return False

    def list_registered_faces(self):
        """List all registered faces with detailed info"""
        if not self.known_face_encodings:
            print("No faces registered yet")
        else:
            print(f"\nRegistered faces ({len(self.known_face_encodings)}):")
            print("-" * 40)
            for i, (name, encodings) in enumerate(self.known_face_encodings.items(), 1):
                metadata = self.face_metadata.get(name, {})
                avg_quality = metadata.get('avg_quality', 0)
                num_samples = len(encodings)

                print(f"{i}. {name}")
                print(f"   Samples: {num_samples}")
                print(f"   Quality: {avg_quality:.2f}")
                if 'capture_time' in metadata:
                    capture_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(metadata['capture_time']))
                    print(f"   Date: {capture_date}")
                print()

    def delete_face(self, name):
        """Delete a registered face"""
        if name in self.known_face_encodings:
            del self.known_face_encodings[name]
            if name in self.face_metadata:
                del self.face_metadata[name]

            image_path = os.path.join(self.faces_dir, f"{name}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)

            self.save_encodings()
            print(f"✅ Face '{name}' deleted successfully!")
        else:
            print(f"❌ Face '{name}' not found!")

    def update_face(self, name):
        """Update an existing face with new samples"""
        if name not in self.known_face_encodings:
            print(f"❌ Face '{name}' not found!")
            return False

        print(f"🔄 Updating face data for: {name}")
        old_sample_count = len(self.known_face_encodings[name])

        old_encodings = self.known_face_encodings[name].copy()
        old_metadata = self.face_metadata[name].copy()
        del self.known_face_encodings[name]
        del self.face_metadata[name]

        success = self.capture_multiple_encodings(name)

        if not success:
            self.known_face_encodings[name] = old_encodings
            self.face_metadata[name] = old_metadata
            print(f"❌ Update failed. Restored previous data for {name}")
            return False

        new_sample_count = len(self.known_face_encodings[name])
        print(f"✅ Updated {name}: {old_sample_count} → {new_sample_count} samples")
        return True

def main():
    face_reg = AdvancedFaceRegistration()

    while True:
        print("\n" + "="*40)
        print("GateGuardian Face Registration")
        print("="*40)
        print("1. Register new face")
        print("2. List registered faces")
        print("3. Delete registered face")
        print("4. Update existing face")
        print("5. Exit")
        print("="*40)

        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            name = input("Enter name for the new face: ").strip()
            if name:
                try:
                    face_reg.capture_multiple_encodings(name)
                except Exception as e:
                    print(f"Registration failed: {e}")
                    face_reg.cleanup_windows()
                finally:
                    # Ensure cleanup always happens
                    face_reg.cleanup_windows()
            else:
                print("Please enter a valid name!")

        elif choice == '2':
            face_reg.list_registered_faces()

        elif choice == '3':
            face_reg.list_registered_faces()
            if face_reg.known_face_encodings:
                name = input("Enter name to delete: ").strip()
                if name:
                    face_reg.delete_face(name)

        elif choice == '4':
            face_reg.list_registered_faces()
            if face_reg.known_face_encodings:
                name = input("Enter name to update: ").strip()
                if name:
                    face_reg.update_face(name)

        elif choice == '5':
            print("Goodbye!")
            # Ensure all windows are closed
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break

        else:
            print("Invalid choice! Please enter 1-5")

if __name__ == "__main__":
    main()
