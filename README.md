# <div align="center">Covid-Detection</div>

# Problem Statement
Covid-19, we all know was a very dangerous disease and the testing of it required proper medical procedure which was time consuming. The report requires minimum 1 day to be produce back to the patient.

# Solution
The Machine Learning/Deep Learning is used to predict the covid Detection using the x-ray Images of patient's chest. Dataset is available which contains x-ray images with Covid and without Covid. Deep Learning Model is applied on it and predict the Covid for patient. It is solving the problem of providing covid report quickly with high accuracy of 95.5%.

# Tech Stack Used
<ul>
<li>Numpy</li>
<li>Pandas</li>
<li>Matplotlib</li>
<li>os</li>
<li>Conv2D Keras Layer</li>
<li>MaxPool2D Keras Layer</li>
<li>Dropout keras Layer</li>
<li>Dense Keras Layer</li>
<li>sklearn</li>
</ul>

# Model Structure
<pre>
Input (image - 224 x 224 x 3)
  |
  |
Conv2D (32 layers , (3 x 3 kernel size))
  |
  |
Conv2D (64 layers, (3 x 3 kernel size))
  |
  |
MaxPool2D (2 x 2)
  |
  |
Dropout (0.25)
  |
  |
Conv2D (64 layers, (3 x 3 kernel size))
  |
  |
MaxPool2D (2 x 2)
  |
  |
Dropout (0.25)
  |
  |
Conv2D (128 layers, (3 x 3 kernel size))
  |
  |
MaxPool2D (2 x 2)
  |
  |
Dropout (0.25)
  |
  |
Flatten
  |
  |
Dense (64)
  |
  |
Dropout (0.5)
  |
  |
Dense (1)
(output)
</pre>

# External Links
