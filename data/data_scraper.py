import pandas as pd
import yfinance as yf
from dateutil import parser
from dateutil.relativedelta import relativedelta
import time
import sys
import os
import re


def get_open_prices(ticker_symbol, start_date, end_date, output_csv):
    '''
    Scrape numerical features of the ticker specified by ticker_symbol from start date to end date,
    and store the scraped data into csv file specified by output_csv
    '''
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            print(f"No data available for {ticker_symbol} for the given date range.")
            return
        # save scraped data to csv file
        data.to_csv(output_csv)
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")


def fetch_shares(ticker, start_date, end_date):
    '''
    Scrape and return shares outstanding information of the ticker specified by ticker_symbol from start date to end date
    '''
    ticker_object = yf.Ticker(ticker)
    return ticker_object.get_shares_full(start=start_date, end=end_date)



if __name__ == '__main__':
    #################################################################
    ##### Part I: for each ticker, scape its numerical features #####
    #################################################################

    # read data and store information of tickers into list
    data_2015 = pd.read_csv('headline/filtered_headlines_2015_error_removed_v2.csv')
    tickers_2015 = data_2015['Stock_symbol'].unique()
    tickers_list_2015 = sorted(tickers_2015.tolist())

    # dictionary that stores minimal and maximal date of each ticker's headlines
    tickers_date_dict_2015 = {}
    for ticker in tickers_list_2015:
        # filter all the rows of the current ticker
        data_ticker = data_2015[data_2015['Stock_symbol'] == ticker]
        # find the earlist and latest date of the headlines of the current ticker
        min_date = data_ticker['Date'].min()
        max_date = data_ticker['Date'].max()
        tickers_date_dict_2015[ticker] = {'min_date': min_date, 'max_date': max_date}

    # scraped csv files are stored in folder "/stock/raw"
    with open("/stock/raw/scrape_output.log", "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        for ticker in tickers_list_2015:
            min_date = tickers_date_dict_2015[ticker]['min_date']
            max_date = tickers_date_dict_2015[ticker]['max_date']
            # one month before the minimum date
            min_date = parser.parse(min_date)
            min_date_one_month_before = min_date - relativedelta(months=1)
            min_date_one_month_before = str(min_date_one_month_before)
            # one day after the maximum date
            max_date = parser.parse(max_date)
            max_date_one_day_after = max_date + relativedelta(days=1)
            max_date_one_day_after = str(max_date_one_day_after)
            # scrape data
            get_open_prices(ticker,
                            min_date_one_month_before[0 : 10],
                            max_date_one_day_after[0 : 10],
                            '/stock/raw/' + ticker + '_numerical_features.csv')
            time.sleep(1)
        sys.stdout = original_stdout
    print('finish scraping raw numerical features, log information in file /stock/raw/scrape_output.log')


    ########################################################################
    ##### Part II: for each ticker, add shares outstanding information #####
    ########################################################################

    # divide tickers into batches
    batch_size = 50
    batches = [tickers_list_2015[i:i+batch_size] for i in range(0, len(tickers_list_2015), batch_size)]

    for batch_index, batch in enumerate(batches):
        # new folder "/stock/raw/" to store csv files with numerical features with shares outstanding
        with open("/stock/raw/scrape_output_withSO" + str(batch_index) + ".log", "w", encoding="utf-8") as f:
            original_stdout = sys.stdout
            sys.stdout = f
            print(f"Processing batch {batch_index+1}/{len(batches)}:")
            for ticker in batch:
                try:
                    data = pd.read_csv('/stock/raw/' + ticker + '_numerical_features.csv')
                    start_date = data["Date"].iloc[0][0:10]
                    end_date = data["Date"].iloc[-1][0:10]
                    history_so = fetch_shares(ticker, start_date, end_date)
                    # add history_so as a column of data
                    data['Date'] = pd.to_datetime(data['Date'])
                    history_so.index = pd.to_datetime(history_so.index)
                    history_so_unique = history_so.groupby(history_so.index).last()
                    data['SharesOutstanding'] = data['Date'].map(history_so_unique)
                    data['SharesOutstanding'] = data['SharesOutstanding'].bfill()
                    data['SharesOutstanding'] = data['SharesOutstanding'].ffill()
                    data.to_csv('/stock/raw/' + ticker + '_numerical_features_withSO.csv', index=True, encoding='utf-8')
                except Exception as e:
                    print(f"Error fetching raw shares outstanding for {ticker}: {e}")
                time.sleep(2)
            print(f"Batch {batch_index+1} completed")
            time.sleep(100)
            sys.stdout = original_stdout


    ########################################################################
    ##### Part III: correctness check and problematic tickers analysis #####
    ########################################################################

    # use regular expression to find tickers with error stored in scrape_output_withSO{i}.log
    tickers_none_history_error = []
    tickers_none_existence_error = []
    for i in range(len(batches)):
        filename = f"/stock/raw/scrape_output_withSO{i}.log"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.search(r'(?<=for\s)([A-Z]+)(?=:)', line)
                    if match:
                        ticker = match.group(1)
                        if "[Errno 2]" in line:
                            tickers_none_existence_error.append(ticker)
                        else:
                            tickers_none_history_error.append(ticker)
        else:
            print(f"file {filename} doesn't exist")
    tickers_none_history_error = sorted(list(set(tickers_none_history_error)))
    tickers_none_existence_error = sorted(list(set(tickers_none_existence_error)))

    assert tickers_none_existence_error == [], "Every ticker in headlines file must has corresponding numerical features!"

    # compute the proportion of headlines of tickers without shares outstanding information
    count = 0
    for ticker in tickers_none_history_error:
        data_ticker = data_2015[data_2015['Stock_symbol'] == ticker]
        count += len(data_ticker)
    proportation = count / len(data_2015)
    print(f"Proportation of headlines of tickers without shares outstanding information is: {proportation}.")