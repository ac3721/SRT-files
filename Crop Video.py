import cv2
import numpy as np

folder = 'Input Video'
video_name = "Pilot.mp4"
video_path = folder + '/' + video_name
cap = cv2.VideoCapture(video_path)

templ = cv2.imread("Template.png", cv2.IMREAD_COLOR)
# mask = cv2.imread("template_mask.png", cv2.IMREAD_GRAYSCALE)  # Optional mask to ignore text

th, tw = templ.shape[:2]  # Template height, width
fps = cap.get(cv2.CAP_PROP_FPS)

# SETUP VIDEO WRITER BEFORE LOOP (size matches template)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_from_match.mp4", fourcc, fps, (tw, th))

# Search state
crop_pos = None  # Will store (x, y) when template found
found_frame = -1
frame_idx = 0

print("Scanning for template...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if crop_pos is None:  # Still searching for first match
        res = cv2.matchTemplate(frame, templ, cv2.TM_CCORR_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.8:  # Adjust threshold as needed
            crop_pos = max_loc
            found_frame = frame_idx
            print(f"✓ Template found at frame {frame_idx} (sec: {frame_idx/fps:.2f})")
    
    # Crop and write if position known (starts from match frame)
    if crop_pos is not None:
        x, y = crop_pos
        crop = frame[y:y+th, x:x+tw]
        
        # Safety check: only write valid crops
        if crop.shape[0] == th and crop.shape[1] == tw:
            out.write(crop)
    
    frame_idx += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()