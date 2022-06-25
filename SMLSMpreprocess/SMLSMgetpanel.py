import wrds
from datetime import datetime, timedelta, date
import statsmodels.formula.api as sm
import pandas as pd 
import holidays
import pandas_market_calendars as mcal

#The following code extracts variables used in the panel data regression analysis from WRDS

db = wrds.Connection(wrds_username='YOUR USERNAME')

#Top 100 companies with regards to news frequency in the years between 2015 and 2019
#Excluded: GLD, SPX, DAX, USD, QQQ, SQ, SNAP, FIT, JCP, FRA, SHOP, AA, PYPL, VRX, ABX, UTX, SPY, AGN, CELG, APC
tickers = ['AAPL', 'AMZN', 'TSLA', 'FB', 'BA', 'NFLX', 'DIS', 'EFX', 'BAC', 'INTC', 'F', 'GE', 'GM', 
 'MSFT', 'SBUX', 'AIR', 'AAL', 'IBM', 'JPM', 'CMG', 'WFC', 'C', 'TWTR', 'WMT', 'MCD', 'AMD', 'NVDA', 'JNJ', 'GS', 
 'BABA', 'CAT', 'MU', 'CSCO', 'XOM', 'CVX', 'BP', 'GOOGL', 'GPRO', 'COST', 'HD', 'NKE', 'KO', 
 'AXP', 'TGT', 'ATVI', 'CMCSA', 'DAL', 'LMT', 'T', 'ABBV', 'PFE', 'GILD', 'ADBE', 'CRM', 'VZ', 'AVGO', 'BX', 
 'LULU', 'BLK', 'UNH', 'KMI', 'BBY', 'PG', 'AGI', 'AMAT', 'MRK', 'M', 'BIDU', 'QCOM', 'FDX', 'AMGN', 
 'BMY', 'ORCL', 'BHP', 'MA', 'KR', 'MO', 'GME', 'PM', 'CHK', 'MMM', 'BBBY', 'COP', 'IRBT', 'MS', 
 'FCX', 'HAL', 'HPQ', 'UAL','JWN', 'CVS', 'V', 'EA', 'STZ', 'GLW', 'ADP', 'AZN', "EBAY", 'ACN', 'PEP']

holidaysUS = holidays.US()
nyse = mcal.get_calendar('NYSE')
stock_holidays = nyse.holidays()

stock_holidays = list(pd.to_datetime(stock_holidays.holidays))
stock_holidays = [x.date() for x in stock_holidays]

#Function to check whether date is on a non-trading day. If so, then return next trading day.
def check_trading_dayandhour(day):
    trading_day = day
    while trading_day.weekday() in holidays.WEEKEND or trading_day in stock_holidays:
        trading_day += timedelta(1)
    return trading_day



#Get all dates between 2015 and 2019
start_date = date(2015, 1, 1)
end_date = date(2019, 12, 31)
delta = end_date - start_date
dates = []

for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    dates.append(day)

data_fill = []
i = 0

for tick in tickers[:25]:

    for date in dates:

        date_adj = check_trading_dayandhour(date)

        # Substract days from datetime variable to get start and end date in API request
        day0 = date_adj - timedelta(0)
        dayminus1 = date_adj - timedelta(1)
        dayminus252 = date_adj - timedelta(252)
        dayminus6 = date_adj - timedelta(6)

        # Convert to proper time format for wrds request
        day0 = day0.strftime("%m/%d/%Y")
        dayminus1 = dayminus1.strftime("%m/%d/%Y")
        dayminus6 = dayminus6.strftime("%m/%d/%Y")
        dayminus252 = dayminus252.strftime("%m/%d/%Y")

        permno = db.raw_sql("""select permno, date
                                                        from crsp.dse
                                                        where TICKER = '{}'
                                                        and date  <= '{}'""".format(tick, day0))
        

        #Get latest permno
        permno["date"] = pd.to_datetime(permno["date"], format = "%Y/%m/%d")
        permno = permno.sort_values(by = "date")
        permno = permno["permno"].iloc[-1]
        
        ########## Following code uses wrds database to extract data for control variable: share turnover and nasdaq dummy ##########

        # data query to get control variable share turnover: volume within (-252,-6) divided by shares outstanding on 0
        voldata = db.raw_sql("""select vol 
                                                    from crsp.dsf
                                                    where permno = {}
                                                    and date between '{}' and '{}'""".format(permno, dayminus252,
                                                                                             dayminus6))

        shrout_m1 = db.raw_sql("""select shrout, date
                                                    from crsp.dsf
                                                    where permno = {}
                                                    and date between '{}' and '{}'""".format(permno, dayminus6, day0))
        
        issuno = db.raw_sql("""select issuno
                                                    from crsp.dsf
                                                    where permno = {}
                                                    and date between '{}' and '{}'""".format(permno, dayminus252, day0))    
                                            
        if int(issuno["issuno"].iloc[-1]) != 0:
            issuno_i = 1
        else:
            issuno_i = 0

        shareturnover = (int(voldata["vol"].sum()) / (int(shrout_m1["shrout"].iloc[-1]) * 1000))

        ########## Following code uses wrds database to extract data for control variable: size  ##########

        # Size defined as price * shrout on date -1
        price_and_shroutm1 = db.raw_sql("""select prc, shrout
                                                            from crsp.dsf
                                                            where permno = {}
                                                            and date between '{}' and '{}'""".format(permno, dayminus1,day0))
        

        size = price_and_shroutm1["prc"][0] * int(price_and_shroutm1["shrout"][0])


        ########## Following code uses wrds database to extract data for control variable: BTM  ##########

        dayplus5years = date + timedelta(1825)
        dayplus5years = dayplus5years.strftime("%m/%d/%Y")

        assets_and_equity = db.raw_sql("""select tic, datadate, seq
                                                            from comp.funda
                                                            where tic = '{}'
                                                            and datadate between '{}' and '{}'""".format(tick,
                                                                                                         day0,
                                                                                                         dayplus5years))
        assets_and_equity = assets_and_equity.dropna()

        # BTM defined as book value of total stockholder equity divided by MVE

        btm = (assets_and_equity["seq"].iloc[0] / (size / 1000))

        ########## Following code uses wrds database to extract data for control variable: pref_alpha  ##########

        ret = db.raw_sql("""select date, ret 
                                                                        from crsp.dsf
                                                                        where permno = {}
                                                                        and date between '{}' and '{}'""".format(permno,
                                                                                                                 dayminus252,
                                                                                                                 dayminus6))

        ff = db.raw_sql("""select date, mktrf, smb, hml, rf
                                                                        from ff.factors_daily
                                                                        where date between '{}' and '{}'""".format(
            dayminus252,
            dayminus6))

        # Change fetched data to pandas dataframe
        ff = pd.DataFrame(ff, columns=["date", "mktrf", "smb", "hml", "rf"])
        ret = pd.DataFrame(ret, columns=["date", "ret"])

        # Merge the two dataframes on date and drop NAs so only dates with no missing values are included in the regression
        data_reg = ret.merge(ff,
                             on="date",
                             how="left")

        data_reg = data_reg[data_reg.notna()]

        # Substract risk free rate from daily return to get excess return
        data_reg["excess_return"] = data_reg["ret"] - data_reg["rf"]
        data_reg = data_reg.drop(["rf", "date"], axis=1)

        if (len(data_reg)) >= 60:
          model = sm.ols("excess_return ~ mktrf + smb + hml", data=data_reg).fit()
          pre_falpha = model.params["Intercept"]
        else:
          raise ValueError("Data incomplete")

        ########## Following code uses wrds database to extract data for dependent variable: excess returns  ##########
        # Excess return defined as 1 day period buy and hold return minus value-weighted CRSP return in same period


        # Fetch prices and returns for given buy and hold period
        exc_ret = db.raw_sql("""select date, prc, ret
                                                            from crsp.dsf
                                                            where permno = {}
                                                            and date = '{}'""".format(permno,day0))

        vw_ret = db.raw_sql("""select date, vwretd
                                                            from crsp.dsi
                                                            where date = '{}'""".format(day0))
        

        # Get abnormal returns

        return_i = exc_ret["ret"][0]
        vw_return_i = vw_ret["vwretd"][0]

        ar = return_i - vw_return_i

        # Also add normal return for possible FF3 regression.

        data_fill.append([date, tick, issuno_i, shareturnover,
                      size, btm, pre_falpha,
                      ar, return_i])
        
        
#References:
#Wharton Research Data Services. https://wrds-www.wharton.upenn.edu/. Accessed: 2022-05-12, accessed on 12.05.2022

    
