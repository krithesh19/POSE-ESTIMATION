import cv2
import mediapipe as mp

# Initialize the MediaPipe Hands and Pose models
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Import the mediapipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
# cap = cv2.VideoCapture('Dinesh Karthik hits 22 runs off Rubel Hossain - 19th over of Nidahas Trophy Final.mp4')


cap = cv2.VideoCapture('ASSETS\VIDEOS\Top 10 2011 Cricket World Cup catches.mp4')



# Initialize the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Initialize the MediaPipe Hands and Pose models
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:

    # Loop through each frame of the video
    while cap.isOpened():

        # Read a frame from the video
        ret, frame = cap.read()

        # If we've reached the end of the video, break out of the loop
        if not ret:
            break

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the hand tracking and pose estimation models on the frame
        hand_results = hands.process(frame)
        pose_results = pose.process(frame)

        # If hands were detected, draw the hand landmarks on the frame
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # If a person was detected, draw the pose landmarks on the frame
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert the frame back to BGR for display and writing to the output video
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Write the frame to the output video
        out.write(frame)

        # If the 'q' key is pressed, break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and output video writer
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()
