import requests
from datetime import datetime

# 获取用户的地理位置
def get_location():
    response = requests.get('https://ipinfo.io/json')
    data = response.json()
    city = data.get('city')
    region = data.get('region')
    return city, region

# 获取当前时间和星期几
def get_current_time_info():
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    weekday = now.strftime("%A")  # 星期几
    return current_time, weekday

def get_all_info_str(): # 有问题
    try:
        city, region = get_location()
        current_time, weekday = get_current_time_info()
        print(f"当前城市: {city}, 省份: {region}\n当前时间: {current_time}\n今天是: {weekday}")
        return f"当前城市: {city}, 省份: {region}\n当前时间: {current_time}\n今天是: {weekday}"
    except Exception as e:
        return f"获取信息时发生错误: {e}"
# 主程序
if __name__ == "__main__":
    city, region = get_location()
    current_time, weekday = get_current_time_info()
    
    print(f"当前城市: {city}, 省份: {region}")
    print(f"当前时间: {current_time}")
    print(f"今天是: {weekday}")
