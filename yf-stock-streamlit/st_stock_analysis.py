import pandas as pd
import pandas_ta as ta
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
from datetime import date, timedelta
import yfinance as yf
import numpy as np
import mplfinance as mpf
import seaborn as sns
import altair as alt
import streamlit as st

st.title("Real-time Stock Price Data")
stock_code = st.text_input("Yahoo Finance Stock Code:", 'ITMG.JK')

# basic configuration
BB_length = 20
BB_mult = 1.5
KC_length = 20
KC_mult = 1.5

stock = yf.Ticker(stock_code)
stock = stock.history(period='max')

year_list = tuple(stock.reset_index().Date.dt.year.unique())
year = str(st.selectbox("Year:", year_list, index=len(year_list)-1))

stock_BB = stock.copy()
stock_BB['MA'] = stock_BB['Close'].rolling(BB_length).mean()
stock_BB['STD'] = stock_BB['Close'].rolling(BB_length).std(ddof=0)
stock_BB['BOLU'] = stock_BB['MA'] + BB_mult*stock_BB['STD']
stock_BB['BOLD'] = stock_BB['MA'] - BB_mult*stock_BB['STD']
stock_BB = stock_BB.loc[year]

stock_KC = stock.copy()
stock_KC['MA'] = stock_KC['Close'].rolling(KC_length).mean()
stock_KC['high_low'] = stock['High'] - stock['Low']
stock_KC['high_closep'] = np.abs(stock['High'] - stock['Close'].shift())
stock_KC['low_closep'] = np.abs(stock['Low'] - stock['Close'].shift())
stock_KC['true_ranges'] = np.max(stock_KC[['high_low', 'high_closep', 'low_closep']], axis=1)
stock_KC['ATR'] = stock_KC['true_ranges'] .rolling(KC_length).mean()
stock_KC['KCU'] = stock_KC['MA'] + KC_mult*stock_KC['ATR']
stock_KC['KCD'] = stock_KC['MA'] - KC_mult*stock_KC['ATR']
stock_KC = stock_KC.loc[year]

stock_SQUEEZE = stock_BB.reset_index().merge(stock_KC).set_index('Date')[['High', 'Low', 'Open', 'Close', 'MA', 'BOLU', 'BOLD', 'KCU', 'KCD']]
stock_SQUEEZE['Squeeze On'] = np.where((stock_SQUEEZE['BOLD'] > stock_SQUEEZE['KCD']) & (stock_SQUEEZE['BOLU'] < stock_SQUEEZE['KCU']), stock_SQUEEZE['Low'], np.nan)
stock_SQUEEZE['Squeeze Off'] = np.where((stock_SQUEEZE['BOLD'] < stock_SQUEEZE['KCD']) & (stock_SQUEEZE['BOLU'] > stock_SQUEEZE['KCU']), stock_SQUEEZE['High'], np.nan)
stock_SQUEEZE.ta.cti(lazybear=True, append=True)
stock_SQUEEZE['SQZ_GATOT'] = np.where((stock_SQUEEZE['BOLD'] < stock_SQUEEZE['KCD']) 
                                                        & (stock_SQUEEZE['BOLU'] > stock_SQUEEZE['KCU']) 
                                                        & (stock_SQUEEZE['Close'] > stock_SQUEEZE['BOLU'])
                                                        & (stock_SQUEEZE['Open'] < stock_SQUEEZE['BOLU'])
                                                        & (stock_SQUEEZE['CTI_12'] > 0), 
                                                         stock_SQUEEZE['High'], np.nan)
sqz_on = (~(stock_SQUEEZE['Squeeze On'] > 0)*1).replace(1, np.NaN)
sqz_off = (~(stock_SQUEEZE['Squeeze Off'] > 0)*1).replace(1, np.NaN)

linreg_length = st.slider("Linear Regression Length:", min_value=2, max_value=len(stock_SQUEEZE), value=int(len(stock_SQUEEZE)/2), step=1)

def linreg(stock, length, offset):
    x = np.arange(1, len(stock)+1)[-length:]
    y = stock['Close'][-length:].values
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    print(f'm = {m} & c = {c}')
    print(m*(x[-1]-offset)+c)
    return m, c

m, c = linreg(stock_SQUEEZE, linreg_length, 0)
intercept = (m*(len(stock_SQUEEZE))+c) - 500*m
x1 = len(stock_SQUEEZE)-linreg_length
x2 = len(stock_SQUEEZE)
y1 = m*(len(stock_SQUEEZE)-linreg_length)+c
y2 = m*(len(stock_SQUEEZE))+c

deviationSum = 0.0
for i in range(0, linreg_length):
    deviationSum += pow(stock_SQUEEZE['Close'][len(stock_SQUEEZE)-1-i]-(m*(len(stock_SQUEEZE)-i)+c), 2)  
deviation = np.sqrt(deviationSum/(linreg_length))

value_X = stock_SQUEEZE.index[-linreg_length:]
value_Y = stock_SQUEEZE['Close'][-linreg_length:].values

index=[stock_SQUEEZE.iloc[x1:x2].index[0], stock_SQUEEZE.iloc[x1:x2].index[-1]]
stock_SQUEEZE['linreg_y'] = pd.Series(data=[y1,y2], index=index)
stock_SQUEEZE['linreg_y_plus_STD'] = pd.Series(data=[y1+2*deviation, y2+2*deviation], index=index)
stock_SQUEEZE['linreg_y_min_STD'] = pd.Series(data=[y1-2*deviation, y2-2*deviation], index=index)

index_linreg = [stock_SQUEEZE.iloc[x1:x2].index[0], stock_SQUEEZE.iloc[x1:x2].index[-1]]
two_points_1 = list(zip(index_linreg, [y1-2*deviation, y2-2*deviation]))
two_points_2 = list(zip(index_linreg, [y1, y2]))
two_points_3 = list(zip(index_linreg, [y1+2*deviation, y2+2*deviation]))

BB = [mpf.make_addplot(stock_BB[['BOLD', 'BOLU']], linestyle='-.'),
      mpf.make_addplot(stock_BB[['MA']], alpha=0.5)
     ]

KC = [mpf.make_addplot(stock_KC[['KCD', 'KCU']], linestyle='--'),
      mpf.make_addplot(stock_KC[['MA']], alpha=0.5)
     ]

SQZ_GATOT = [mpf.make_addplot(stock_SQUEEZE['SQZ_GATOT'].values, type='scatter', marker='v', color='orange'),
            ] if stock_SQUEEZE['SQZ_GATOT'].notna().sum() > 0 else []

squeeze = [mpf.make_addplot(stock_SQUEEZE['CTI_12'],panel=2,color='g',type='bar', width=0.75, ylabel='Squeeze (LB)'),
           mpf.make_addplot(sqz_on,panel=2,type='scatter', markersize=2, color='blue'),
           mpf.make_addplot(sqz_off,panel=2,type='scatter', markersize=2, color='aqua')
          ]

fig, axlist= mpf.plot(stock_BB, type='candle', volume=True, figsize=(13,8), panel_ratios=(1, 0.2), addplot=BB+squeeze+SQZ_GATOT, returnfig=True,  style='sas',
                      alines=dict(alines=[two_points_1, two_points_2, two_points_3], colors=['red', 'aqua', 'lime'], linewidths=0.5, alpha=0.5))
axlist[0].fill_between([x for x in range(len(stock_BB.index))], stock_BB['BOLD'], stock_BB['BOLU'], facecolor='orange', alpha=0.1)
# axlist[0].legend(['Bollinger Lower', 'Bollinger Upper', 'Mid Bollinger', 'SQZ Gatot Signal', 'a'])
plt.show()

st.pyplot(fig)

## Create Altair Chart

source = stock_SQUEEZE.copy().reset_index().reset_index()

open_close_color = alt.condition("datum.Open <= datum.Close",
                                 alt.value("#66CC00"),
                                 alt.value("#ae1325"))

base = alt.Chart(source).encode(
    alt.X('index:Q',
          axis=alt.Axis(
#               format='%m/%d',
              labelAngle=-45,
              title='Tick Nos'
          )
    ),
)

rule = base.mark_rule().encode(
    alt.Y(
        'Low:Q',
        title='Price',
        scale=alt.Scale(zero=False),
#        
    ),
    alt.Y2('High:Q'),
    color=open_close_color
)

BB_UP = base.mark_line(strokeWidth=1, opacity=0.3, color='lime', strokeDash=[10, 5]).encode(
    alt.Y('BOLU:Q')
)

BB_LOW = base.mark_line(strokeWidth=1, opacity=0.3, color='red', strokeDash=[10, 5]).encode(
    alt.Y('BOLD:Q')
)

BB = base.mark_area( 
    color='green',
    opacity=0.1
).encode(
    alt.Y('BOLD:Q'),
    alt.Y2('BOLU:Q')
)

MA_20 = base.mark_line(strokeWidth=1, opacity=0.3, color='blue').encode(
    alt.Y('MA:Q')
)

linreg_y = alt.Chart(source[source.linreg_y.notna()]).mark_line(strokeWidth=2, color='aqua', opacity=0.5).encode(
    x = ('index:Q'),
    y = ('linreg_y:Q')
)

linreg_y_plus_STD = alt.Chart(source[source.linreg_y.notna()]).mark_line(strokeWidth=2, color='lime', opacity=0.5).encode(
    x = ('index:Q'),
    y = ('linreg_y_plus_STD:Q')
)

linreg_y_min_STD = alt.Chart(source[source.linreg_y.notna()]).mark_line(strokeWidth=2, color='red', opacity=0.5).encode(
    x = ('index:Q'),
    y = ('linreg_y_min_STD:Q')
)

bar = base.mark_rule(strokeWidth=4).encode(
    alt.Y('Open:Q'),
    alt.Y2('Close:Q'),
    color=open_close_color,
    tooltip=['Date:T','Open:Q', 'Close:Q', 'Low:Q', 'High:Q', 'MA:Q']
).properties(
    width=700,
    height=500,
    title=f"{stock_code} Stock Price").interactive()

st.write(bar + rule + BB + MA_20 + BB_UP + BB_LOW + linreg_y + linreg_y_plus_STD + linreg_y_min_STD)

## Dividends Chart

stock_yield = stock.drop(['Open', 'High', 'Low', 'Volume'], axis=1).resample('Y').agg({
    'Close' : 'max',
    'Dividends' : 'sum'
})

stock_yield = stock_yield[stock_yield['Dividends'] != 0].reset_index()
stock_yield['Date'] = np.datetime_as_string(stock_yield['Date'], unit='Y')
stock_yield['Yield (%)'] = stock_yield['Dividends']/stock_yield['Close']*100

def show_dividends_yield():
    fig_2 = plt.figure(figsize=(13,8))
    clrs = ['blue' if (x == year) else 'aqua' for x in list(stock_yield.Date.values)]
    splot = sns.barplot(x='Date', y='Yield (%)', data=stock_yield, palette=clrs, alpha=0.5)

    for p in splot.patches:
        if p.get_height() == stock_yield[stock_yield.Date == year]['Yield (%)'].values[0]:
            splot.annotate(format(p.get_height(), '.1f') + "%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 5), 
                        fontsize=10,
                        weight='bold',
                        textcoords = 'offset points')
        else:
            splot.annotate(format(p.get_height(), '.1f') + "%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 5), 
                        fontsize=8,
                        textcoords = 'offset points')

        
    plt.xlabel('Year')
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylabel('Dividend Yield (%)')
    st.pyplot(fig_2)

if stock_yield.shape[0] > 1:
    show_dividends_yield() 
else: 
    st.write('No Dividends data available for', stock_code)