import pandas as pd
import numpy as np
import math
import csv

HOUR_THRESHOLD = 60
DAY_THRESHOLD = 9*24
WEEK_THRESHOLD = 5*7*24

def read_data():
    df = pd.read_excel('Dataset_PredictionModel.xlsx')
    global data,date
    # print the column names
    # print(df.columns)
    # get the values for a given column
    df = df.fillna(df.mean())
    values = df.values
    date = values[:, 0]
    data = values[:, 1:]
    m, n = data.shape
    # print(type(data[-1,-1]))
    return data.reshape((m*n,1))

def quy(month):
    if (month <= 3):
        return 1
    if (month <= 6):
        return 2
    if (month <= 9):
        return 3
    return 4

def data_in_hour():
    global data,date, hour_in_day
    all_data = []
    hour_in_day = []
    m,n = data.shape
    for i in range(m):
        tg = date[i].split(" ")
        for j in range(n):
            item = []
            item.append(data[i][j])
            item.append(int(tg[1]))
            item.append(int(tg[2]))
            q = quy(int(tg[1]))
            item.append(q)
            item.append(j+1)
            item.append( (i%7)+1 )
            hour_in_day.append(i)
            all_data.append(item)
    data_by_hour = np.array(all_data)
    # print(hour_in_day)
    return data_by_hour

def data_in_day():
    global data, date, day_in_week, day_avg
    all_data = []
    day_in_week = []
    day_avg = []
    m, n = data.shape
    for i in range(m):
        tg = date[i].split(" ")
        days = np.sum(data[i,:])
        item = []
        item.append(days)
        item.append(int(tg[1]))
        item.append(int(tg[2]))
        q = quy(int(tg[1]))
        item.append(q)
        item.append((i % 7) +1)
        day_in_week.append(math.floor(i/7))
        day_avg.append(days/24)
        all_data.append(item)
    data_by_day = np.array(all_data)
    # print(day_avg)
    # print(len(day_avg))
    # print(day_in_week)
    return data_by_day

def data_in_week():
    global data, date, week_avg
    all_data = []
    week_avg = []
    m, n = data.shape
    for i in range(0,m,7):
        tg = date[i].split(" ")
        weeks = 0
        for j in range(7):
            try:
                days = np.sum(data[i+j,:])
            except IndexError:
                print("oops %d"%(i+j))
            weeks += days
        item = []
        item.append(weeks)
        item.append(int(tg[1]))
        q = quy(int(tg[1]))
        item.append(int(tg[0]))
        item.append(q)
        all_data.append(item)
        week_avg.append(weeks/7)
    data_by_week = np.array(all_data)
    #print(data_by_week.shape)
    # print(week_avg)
    # print(len(week_avg))
    return data_by_week

def data_in_month():
    global data, date, month_avg, week_in_month
    week_in_month = []
    month_avg = []
    m, n = data.shape
    current_month = int(date[0].split(" ")[1])
    month_sum = 0
    day_sum = 0
    for i in range(m):
        tg = date[i].split(" ")
        month = int(tg[1])
        days = np.sum(data[i, :])
        if (month == current_month):
            month_sum += days
            day_sum += 1
        else:
            month_avg.append(7*month_sum/day_sum)
            month_sum = days
            day_sum = 1
            current_month = month
        if (i%7 == 0):
            week_in_month.append(len(month_avg))
    month_avg.append(month_sum / day_sum)
    # print(month_avg)
    # print(len(month_avg))
    # print(week_in_month)

    # print(data_by_week.shape)


def replace_abnormal_detection_in_hour(data):
    global day_avg, hour_in_day
    num_abnormal = 0
    mean = np.average(data[:,0])
    abnormal_list = []
    print(mean)
    for i in range(data.shape[0]):
        # if (i == 0):
        #     if ( math.fabs(data[i+1][0] - data[i][0]) >= HOUR_THRESHOLD):
        #         data[i][0] = day_avg[hour_in_day[i]]
        #         num_abnormal += 1
        # elif (i == data.shape[0]-1):
        #     if ( math.fabs(data[i][0] - data[i-1][0]) >= HOUR_THRESHOLD):
        #         data[i][0] = day_avg[hour_in_day[i]]
        #         num_abnormal += 1
        # else:
        #     if ( math.fabs(data[i][0] - data[i-1][0]) >= HOUR_THRESHOLD) and ( math.fabs(data[i+1][0] - data[i][0]) >= HOUR_THRESHOLD):
        #         data[i][0] = day_avg[hour_in_day[i]]
        #         num_abnormal += 1
        if (math.fabs(data[i][0] - mean) >= float(HOUR_THRESHOLD)):
            # print(data[i][0])
            abnormal_list.append(float(data[i][0]))
            data[i][0] = day_avg[hour_in_day[i]]
            num_abnormal += 1
    print(num_abnormal)
    with open('abnormal.csv', 'w') as csvfile:
        fieldnames = ['abnormal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for abnormal in abnormal_list:
            writer.writerow({'abnormal': abnormal})
    return data

def replace_abnormal_detection_in_day(data):
    global week_avg, day_in_week
    num_abnormal = 0
    for i in range(data.shape[0]):
        if (i == 0):
            if ( math.fabs(data[i+1][0] - data[i][0]) >= DAY_THRESHOLD):
                data[i][0] = week_avg[day_in_week[i]]
                num_abnormal += 1
        elif (i == data.shape[0]-1):
            if ( math.fabs(data[i][0] - data[i-1][0]) >= DAY_THRESHOLD):
                data[i][0] = week_avg[day_in_week[i]]
                num_abnormal += 1
        else:
            if ( math.fabs(data[i][0] - data[i-1][0]) >= DAY_THRESHOLD) and ( math.fabs(data[i+1][0] - data[i][0]) >= DAY_THRESHOLD):
                data[i][0] = week_avg[day_in_week[i]]
                num_abnormal += 1
    print(num_abnormal)
    return data

def replace_abnormal_detection_in_week(data):
    global month_avg, week_in_month
    num_abnormal = 0
    for i in range(data.shape[0]):
        if (i == 0):
            if ( math.fabs(data[i+1][0] - data[i][0]) >= WEEK_THRESHOLD):
                data[i][0] = month_avg[week_in_month[i]]
                num_abnormal += 1
        elif (i == data.shape[0]-1):
            if ( math.fabs(data[i][0] - data[i-1][0]) >= WEEK_THRESHOLD):
                data[i][0] = month_avg[week_in_month[i]]
                num_abnormal += 1
        else:
            if ( math.fabs(data[i][0] - data[i-1][0]) >= WEEK_THRESHOLD) and ( math.fabs(data[i+1][0] - data[i][0]) >= WEEK_THRESHOLD):
                data[i][0] = month_avg[week_in_month[i]]
                num_abnormal += 1
    print(num_abnormal)
    return data

#data = replace_nan(data)
read_data()
data_in_hour()
data_in_day()
data_in_week()
data_in_month()
#replace_abnormal_detection_in_hour(data_in_hour())
#replace_abnormal_detection_in_day(data_in_day())
#replace_abnormal_detection_in_week(data_in_week())