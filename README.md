# RoboCupHumanoid
In this project, we are aiming to implement [1] and improve it using Conv + LSTM layer for detection and tracking of the soccer ball for <a href="https://www.robocuphumanoid.org/">RoboCup Humanoid League</a>.

### Dataset
data.csv contains information about training images with the following fields:
<ul>
  <li><b>image_file</b> - image location,</li>
  <li><b>width</b> - width of the image,</li>
  <li><b>height</b> - height of the image,</li>
  <li><b>label</b> - label of the object,</li>
  <li><b>xmin</b> - top left x-coordinate of rectangle around object,</li>
  <li><b>ymin</b> - top left y-coordinate of recatngle around object,</li>
  <li><b>xmax</b> - bottom right x-coordinate of recatngle around object,</li>
  <li><b>ymax</b> - bottom right y-coordinate of recatngle around object.</li>
</ul>
<b>Note</b>: One image file may contain different object types of different types.

## References
[1] Fabian Schnekenburger, Manuel Scharffenberg, Michael Wulker, Ulrich Hochberg, Klaus Dorer [*Detection and Localization of Features on a Soccer Field with Feedforward Fully Convolutional Neural Networks (FCNN) for the Adult-Size Humanoid Robot Sweaty*](http://lofarolabs.com/events/robocup/ws17/papers/Humanoids_RoboCup_Workshop_2017_pape_4.pdf)
