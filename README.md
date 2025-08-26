# VFNN
Neural networks for volatility forecasting. Work follows Deep Learning Stock Volatility with Google Domestic Trends (Xiong et al.)

## Table of Contents
- [Plan/Intro](#plan/intro)
- [Implementation](#implementation)
- [Takeaways](#takeaways)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Plan/Intro
In this project I will build a LSTM neural network for volatility forecasting on the S&P 500 using google trends data.
It is interesting that the model does not directly predict the price or the return (difference in price over a period) but the fluctuation in the the daily returns over the next couple of days based on the fluctuations in the previous thirty days and trends data on these days. Additionally, it is fascinating to observe the predictive power of something as simple as a tally of a word's search history over something as complicated as stock trends.

## Implementation
The model was built with PyTorch and trained on trends data from the PyTrends api and financial data from the yfinance api. Most of the work was in the hyperparameter tuning and normalization strategies for the data, as financial data can be quite difficult to work with in machine learning contexts.

## Takeaways
I was very happy with the results, I outperformed the paper on root mean squared error by three orders of magnitude and mean absolute percent error by eleven percent. Although, I suspect this is becuase I spent a lot of time regularizing and tuning my model, whereas the main focus of the paper was on displaying the effectivness of the model architecture and dataset selection in comparison to unsupervised and recursive algorithms.

## Acknowledgments

The ideas in **VFNN** are practically entirely based on the paper:

- [Xiong et al. (2016)](https://arxiv.org/pdf/1512.04916) — *Deep Learning Stock Volatility with Google Domestic Trends*.
  
## License
This project is licensed under the [MIT License](LICENSE).
