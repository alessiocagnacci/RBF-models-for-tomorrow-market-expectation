# RBF Models for Tomorrow Market Expectation 

A systematic, RBF-based trading engine designed to forecast next-day market behavior. 

While originally backtested and optimized for the unique micro-structure of the Japanese stock market (**Nikkei 225**), the architecture is completely **ticker-agnostic**. By simply feeding it a different historical dataset, the model can be deployed on any index, stock, or crypto asset.

## Core Objectives & Predictions
Unlike traditional models that only guess market direction, this project uses Support Vector Machines with Radial Basis Function (RBF) kernels to generate a total view of tomorrow's trading session. For any given day, the model predicts:

1. **Market Trend:** Bullish or Bearish directional probability.
2. **Absolute Volatility:** The expected magnitude of the price movement.
3. **Volatility Regime:** Classification of market risk into distinct zones (e.g., Low, Normal, High, Extreme) to dynamically adjust position sizing and stop-losses.

By combining directional alpha with strict volatility-based risk management, this engine filters out market noise and executes trades only when statistical confidence is high.

Package used: yfinance, pandas, numpy, sklearn, matplotlib

##  Performance (Nikkei 225 Backtest)
The tree-engine architecture has been rigorously backtested on historical data from the Nikkei 225 index. 

Here are the baseline Out-of-Sample error result:
* **Absolute volatility Engine:** 
  Achieved an **RMSE of 2.2** (less than 10% errore rate) in predicting the absolute volatility class. This is a significant result added with **RMSE of 4,2 in tail events**  proving the Gaussian RBF ability to reproduce absolute volatility.
  
* **Volatility Regime Engine:** 
  Achieved an **F1-Score of 0.79** in predicting the "Extreme" volatility class. This is a highly significant result for financial datasets, proving the RBF kernel's ability to isolate rare, portfolio-destroying events despite severe class imbalance.
  
* **Trend Engine:** 
  Achieved a baseline raw directional accuracy of **~55%**. While seemingly modest but in quantitative finance is a highly significant result. 
  
**KAN and MLP** 
  I also tried this two type of engine other than RBF obteining less significant result, by this the chose of using RBF methods.
