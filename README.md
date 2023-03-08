# Object Recognition
Object Recognition in real time for live video with OpenCV

This program matches an object (ideally on some plain white background) and generates a binary image. From that image, object segments are identified and classified based on a few features that are rotation/scale/translation invariant.

This program takes 1 optional argument, which is an integer value for k that is used if the user wants to use k-nearest neighbors instead of the default nearest neighbor for classifying an object. I restrict this value to between 1 and 5, since that was the number of minimum entries for each object I used in my database.

The project expects a video stream for classifying objects.
The database file is called "object_database" with no file extension.
To enter a new object into the database, press the 'n' key to pause the frame and enter the object name into the console. This saves the currently processed feature into the database, which will be loaded the next time the program starts.
