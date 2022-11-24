import datetime

def tuples_to_dict(data):
    res={}
    for key, value in data:
        res[key]=[]
    for key, value in data:
        res[key].append(value)
    for key, value in res.items():
        if(len(value)==1):
            res[key] = value[0]
    return res

def unixtodate(value, format='%d/%m/%Y %H:%M'):
    print(f"value: {value}")
    if(len(value)>0):
        ts = int(value)
        dt = datetime.datetime.fromtimestamp(ts / 1000)
        formatted_time = dt.strftime(format)
        return formatted_time
    else:
        return ""

