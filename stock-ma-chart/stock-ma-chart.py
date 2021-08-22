import matplotlib.pyplot as plt
import yfinance as yf

aaple = yf.Ticker("AAPL")
prices = aaple.history(period='1y')

prices['20d'] = prices['Close'].rolling(20).mean()
prices['50d'] = prices['Close'].rolling(50).mean()

prices[['20d', '50d']].plot()

plt.title('Moving Averages Apple')
