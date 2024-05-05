# Loss-Function-Library
A library of loss functions to use on your supervised learning models.

In machine learning, loss functions serve as a way to determine how accurate your model's predictions(x) are compared to the labeled data(y) to quantify how happy you are with your parameters(W). However, there isn't a single loss function that works for all algorithms. Selecting a loss function for a given problem involves a number of considerations, including the machine learning method of choice, the simplicity of the derivative calculations, and, to some extent, the proportion of outliers in the data set.

Generally speaking, loss functions fall into two main groups based on the kind of learning job we are working on: regression losses and classification losses. In classification, we attempt to forecast the result of a set of finite categorical values, i.e., classifying a big data set of images handwritten digits into one of nine possible categories.Conversely, regression is concerned with forecasting a continuous value, such as a particular floor area, number of rooms, or size of rooms, and predicts the room's price.

# Regression Losses
There are three main forms of regression loss used in deep learning: L<sub>2</sub> Loss (Mean Square Error/Quadratic Loss), L<sub>1</sub> Loss (Mean Absolute Error), and Mean Bias Error.

## Mean Square Error (MSE)/Quadratic Loss/L<sub>2</sub> Loss
Mathematical Formula:
<img><img src='https://miro.medium.com/v2/resize:fit:786/format:webp/1*SGhoeJ_BgcfqU06CmX41rw.png'>

As the name suggests, MSE measures the average of the squared difference between predictions and actual observations. It's only concerned with their average magnitude—not the direction of errors. Squaring, however, results in a heavier penalty for forecasts that differ greatly from actual values compared to less deviant predictions. Additionally, MSE has good mathematical qualities that facilitate gradient computation.

See MSE.py for implimentation of L<sub>2</sub> Loss

## Mean Absolute Error/L<sub>2</sub> Loss
Mathematical Formula:
<img></img src='https://miro.medium.com/v2/resize:fit:786/format:webp/1*piCo0iDgPmESnQkHSwAK6A.png'>