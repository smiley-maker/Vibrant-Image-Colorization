# Vibrant-Image-Colorization
Introduces a new loss function for image colorization that prioritizes saturation over hue in an effort to decrease the effect of ambiguity and to produce vibrant and colorful images.

Image colorization is a challenging task in computer vision as it necessitates the selection of colors that might be ambiguous or challenging to understand from a greyscale input. Traditional methods have utilized deep learning to understand the relationships between pixels in an image and accurately colorize them. However, these approaches suffer from ambiguous objects (such as apples that might be generally green or red) as the loss functions are often based on the mean squred error (MSE) between the predictions and the ground truth colors. In this work, I implemented a new loss function that downweights the importance of hue, while prioritizing saturation differences in an effort to decrease the effect of color uncertainty and to produce more vibrant images. 

I implemented a convolutional neural network (CNN) trained with a custom loss function. The loss function is formulated as follows. The input to the neural network is the luminousity (L) channel of a LAB color space image. The task is to accurately predict the A and B color channel values for each pixel in the image. My loss function calculates the ground truth and predicted hue and saturation from the A,B color channels. Then the saturation component is weighted by some factor. Finally, the differences between the ground truth and predicted hue and saturation values are added to form the overall loss. I also experimented with a kind of "saturated huber loss" which combines the approach discussed above with a standard mean squared error (MSE) loss that is downplayed against the saturation and hue loss. This helps to promote color accuracy while still giving the model room for ambiguity. 

## Initial Results


<img src="./images/300_epochs_pokemon.png" alt="Example colorization after 300 epochs." />

<img src="./images/500_epochs.png" alt="Example colorization after 500 epochs." />

<img src="./images/flamingo.png" alt="Example colorization of a flamingo photograph. " />

<img src="./images/water.png" alt="Example colorization of water photograph." />
