# Drowsiness-detector-in-python
Making use of opencv facial landmarks for blink and drowsiness detection

Face detection is first employed and the eye regions are extracted.

The eye aspect ratio is then calculted with the 6 points of each extracted eye region to determine whether the user blinks or not.

If the eyes are closed for a considerable amount of time, a beeping sound is triggered to alert the user that he or she is sleeping.
