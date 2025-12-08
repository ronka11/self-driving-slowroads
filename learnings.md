# Learnings 
just dumping important stuff here

perception → planning → control

1. Perception: 
This is the process where the camera feed (in slowroads.io, the rendered road) and extract features like:
- Lane lines
- Obstacles
- Road curvature
- Traffic information (if present)
Things to use
- Lane detection models: ENet-SAD, SCNN, or modern lightweight models like UltraFast-Lane-Detection, LSTR, or even YOLOv8-seg for segmentation.
- Depth or semantic maps: (optional) to infer distance to curves or cars.

2. Planning: 
This is the "brain" that decides what to do next. It takes the perception output and decides:
- Keep in lane
- Turn slightly left/right
- Slow down or accelerate
Can implement this via:
- Classical control (PID controller): Uses lane deviation angle and curvature to steer.
- Machine learning-based control: e.g. behavioral cloning (imitation learning) — your model learns directly from human steering data.

3. Control: 
This converts planning decisions into actual game inputs:
- Steer left/right
- Accelerate/brake