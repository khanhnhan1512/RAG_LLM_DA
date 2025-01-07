from datetime import datetime, timedelta  

class DateMapper:  
    def __init__(self, year=2014):  
        self.year = year  
        self.start_date = datetime(year, 1, 1)  
        self.date_to_id_map = {}  
        self.id_to_date_map = {}  
        self._build_mappings()  
    
    def _build_mappings(self):  
        """Xây dựng mapping hai chiều"""  
        current_date = self.start_date  
        for day_id in range(1, 367):  # 366 ngày cho năm nhuận  
            date_str = current_date.strftime('%Y-%m-%d')  
            self.date_to_id_map[date_str] = day_id  
            self.id_to_date_map[day_id] = date_str  
            current_date += timedelta(days=1)  
    
    def date_to_id(self, date_str):  
        """Chuyển đổi ngày thành ID"""  
        try:  
            return self.date_to_id_map[date_str]  
        except KeyError:  
            raise ValueError(f"Invalid date or date not in year {self.year}: {date_str}")  
    
    def id_to_date(self, day_id):  
        """Chuyển đổi ID thành ngày"""  
        try:  
            return self.id_to_date_map[day_id]  
        except KeyError:  
            raise ValueError(f"Invalid ID. Must be between 1 and 366")  
    
    def convert_dates(self, dates):  
        """Chuyển đổi nhiều ngày cùng lúc"""  
        return [self.date_to_id(date) for date in dates]  

# Sử dụng  
mapper = DateMapper(2014)  

# Danh sách ngày cần chuyển đổi  
dates = ['2014-12-31', '2014-12-30', '2014-01-01']  

# Chuyển từ ngày sang ID  
ids = mapper.convert_dates(dates)  
print("Dates to IDs:", ids)  

# Chuyển từ ID sang ngày  
for id in ids:  
    print(f"ID {id} -> Date {mapper.id_to_date(id)}")  

# Tạo dictionary mapping  
date_id_dict = {date: mapper.date_to_id(date) for date in dates}  
print("Mapping dictionary:", date_id_dict)