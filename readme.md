<h1> TASK 1 </h1>

Face detection and liveness detection
Step 1
Detect in a single frame the number of human faces out of the total objects detected.
Step 2
If the number of faces detected in Step 1 is > 1 or = 0, Exit
Step 3
For a single detected face, check the liveness score*
Step 4
Against a preset threshold value, if liveness score in step 3 is below that value declare
success otherwise Exit
*Liveness score will be determined for any change in the face attributes like eye wink/blink,
smile etc. 
