(English version [here](README.md))
# AIVIVN - Dự đoán lưu lượng server 2
Model của mình cho cuộc thi thứ 5 của AIVIVN: [Dự đoán lưu lượng server 2](https://www.aivivn.com/contests/5). 

## Hướng dẫn tạo file submission
Để chạy code, các bạn làm như sau (đã chạy OK trên Python 3.6 và 3.7):
- Download các file dữ liệu cần thiết của cuộc thi (`test_id.csv` và `train.csv`) và để vào thư mục `data`.
- Cài dependencies nếu cần thiết: `pip install -r requirements.txt`.
- Chạy file `main_combined.py`.

## Giới thiệu
*(Lấy từ từ trang web của cuộc thi)*

Một công ty cung cấp nền tảng giải trí cho phép user sử dụng các dịch vụ music, video, live stream, chat, … Hệ thống công ty chia thành các zone theo khu vực địa lý. Để đáp ứng số lượng user ngày càng tăng, công ty muốn dự đoán được tổng bandwidth của mỗi server và số lượng tối đa user truy cập đồng thời vào server trong vòng một tháng tiếp theo để lên kế hoạch hoạt động.

## Data
Tất cả dữ liệu nằm trong thư mục `data`.
### Tập training
Dữ liệu huấn luyện (file `train.csv`) gồm hơn 35 nghìn dòng. Dưới đây là năm dòng đầu tiên.

```csv
UPDATE_TIME,ZONE_CODE,HOUR_ID,BANDWIDTH_TOTAL,MAX_USER
2017-10-01,ZONE01,0,16096.71031272728,212415.0
2017-10-01,ZONE01,1,9374.20790727273,166362.0
2017-10-01,ZONE01,2,5606.225750000003,146370.0
2017-10-01,ZONE01,3,4155.654660909094,141270.0
2017-10-01,ZONE01,4,3253.9785936363623,139689.0
```

- `UPDATE_TIME`: ngày thực hiện lấy dữ liệu
- `HOUR_ID`: giờ thực hiện lấy dữ liệu
- `ZONE_CODE`: mã khu vực
- `BANDWIDTH_TOTAL`: tổng băng thông truy cập tương ứng trong vòng 1 giờ
- `MAX_USER`: số user truy cập đồng thời tối đa trong vòng 1 giờ (là một số tự nhiên)

### Tập Testing 
Dữ liệu kiểm tra (file `test.csv`) bao gồm 2227 dòng có dạng:

```csv
id,UPDATE_TIME,ZONE_CODE,HOUR_ID
0,2019-03-10,ZONE01,0
1,2019-03-10,ZONE01,1
2,2019-03-10,ZONE01,2
3,2019-03-10,ZONE01,3
4,2019-03-10,ZONE01,4
5,2019-03-10,ZONE01,5
```
Trong đó `UPDATE_TIME, HOUR_ID, ZONE_CODE` được định nghĩa như trên, id là mã số tương ứng cho file nộp bài. Các đội chơi cần dự đoán `BANDWIDTH_TOTAL`, và `MAX_USER` cho mỗi dòng.

## Hướng giải quyết
Mình đã thử một vài phương pháp cho bài toán này, các baseline dưới đây đều cho kết quả public LB tốt.
- *Baseline 0 (Moving average)*: Mấu chốt của cách này là tính theo từng zone và từng giờ: Với mỗi zone và mỗi giờ, mình tính moving average của `x` ngày gần nhất trên tập train và dùng kết quả để dự đoán trên tập test cho zone và giờ tương ứng. Xem thêm code ví dụ tại [đây](https://forum.machinelearningcoban.com/t/aivivn-bandwidth-prediction-baseline-with-moving-average/5488).
- *Baseline #1 (Median of medians)*: Tương tự với moving avg/median, nhưng lần này mình tính moving median của nhiều hơn 1 window khác nhau và lấy median của tất cả các median tính được.
- *Baseline #2 - XGBoost* (Đọc thêm về XGBoost ở phần [Tham khảo](#tham-khảo)). Với cả 2 biến target (`bandwidth_total` và `max_user`), mình đều dùng XGBoost làm model duy nhất. Phần chiếm thời gian của mình nhiều nhất là nghiên cứu xem chọn feature thế nào.

Final model của mình (cho kết quả public LB cao nhất) là kết hợp của Baseline #1 và Baseline #2 (`final_prediction = 0.8 * XGBoost + 0.2 * median_of_medians`).

### Missing values
Data có một số thời điểm không có dữ liệu. Mình thêm vào bằng cách lấy dữ liệu cùng ngày cùng giờ của tuần trước (Ví dụ nếu dữ liệu bị mất vào 4h sáng thứ 4 thì mình sẽ lấy dữ liệu 4h sáng thứ 4 tuần trước).

### Features
#### Time features (các đặc trưng về thời gian)
Các feature dưới đây có thể lấy dễ dàng dựa vào 2 cột `update_time` và `hour_id`
- Hour (giờ)
- Day (ngày)
- Week (tuần)
- Day of week (ngày trong tuần)
- Month (tháng)
- Day of year (ngày trong nằm)
- Year (năm)

#### Special event features (Đặc trưng cho những sự kiện 'đặc biệt')
##### Abnormal events (sự kiện bất thường)
Khi vẽ các time series của từng zone lên mình thấy có 2 khoảng thời gian đáng lưu ý, khi mà:
1. `ZONE01`: Total bandwidth và max users giảm đột biến.
2. `ZONE02`: Total bandwidth và max users tăng đột biến.
3. `ZONE03`: Total bandwidth tăng rất nhẹ không đáng kể, max users hơi tăng hơn so với bình thường (độ tăng nhỏ hơn so với `ZONE02`).

Cả 2 khoảng thời gian này đều rơi vào đầu năm (2018 và 2019). Từ đó, mình tạo thêm 2 feature mới là `abnormal_bw` và `abnormal_u` để 'đo' mức độ ảnh hưởng của những khoảng tăng/giảm đột biến này đối với total bandwidth và max users (Lúc đầu mình dùng boolean - `1` cho những ngày đặc biệt, `0` cho những ngày còn lại - nhưng cho kết quả không tốt bằng). 
- Với `ZONE01`, do cả total bandwidth và max users giảm đột biến, mình cho `abnormal_bw` và `abnormal_u` bằng `-1` trong những ngày đặc biệt, `0` cho các ngày khác. 
- Tương tự với `ZONE02`, các biến sẽ lần lượt là `0.8` vào ngày đặc biệt và `0` cho ngày khác. 
- Với `ZONE03`, mình cho `abnormal_bw` bằng `0.2`  do total bandwidth chỉ tăng rất nhẹ (gần như không tăng đột biến), nhưng cho `abnormal_u` bằng `0.6` vì max users tăng rõ rệt hơn.

##### Holiday events (Ngày lễ)
Mình dùng biến boolean `holiday` để ghi nhận các ngày lễ của Việt Nam. Mình có thêm cả 3 ngày từ 23/12/2017 đến 25/12/2017 vào danh sách holidays, do total bandwidth và max user của `ZONE01` tăng đột biến.

#### Zone features (Đặc trưng từng zone)
##### Median (trung vị)
Từ dữ liệu train, mình lấy được median của total bandwidth và max users cho từng zone, với các khoảng thời gian: 1 tháng gần nhất, 3 tháng gần nhất, 6 tháng gần nhất, 1 năm gần nhất:
- `median_user_1m` 
- `median_bw_1m`  
- `median_user_3m` 
- `median_bw_3m` 
- `median_user_6m` 
- `median_bw_6m` 
- `median_user_1y` 
- `median_bw_1y`
- `median_bw_per_user_6m`
- `median_bw_per_user_3m`
- `median_bw_per_user_1m`
- `median_bw_per_user_1y`

##### Autocorrelation features (Đặc trưng tự tương quan)
Với 3 tháng gần nhất trong tập train, mình tính độ tự tương quan cho mỗi zone với các độ trễ (lag) lần lượt là 1,3,7 ngày.

#### Linear regression prediction features
Cuối cùng, mình dùng linear regression để fit tập train (dùng các feature về thời gian, sự kiện đặc biệt và median của zone) và sau đó dùng chính dự đoán của model đó (trên cả tập train và test) để làm feature mới cho XGBoost (`ridge_bw` và `ridge_u`). Mình đã thử 2 model của sklearn là Ridge và Lasso (dùng các thông số mặc định) thì Ridge cho kết quả public tốt hơn nên mình chọn Ridge.


## Kết quả
Final model cho kết quả `sMAPE = 5.12708` trên public LB.

## Mở rộng
Có rất nhiều hướng khác để giải quyết bài toán time series, trong tương lai gần mình sẽ cố gắng thử áp dụng Deep Learning lên data này và so sánh với các phương pháp ML truyền thống. Với ML truyền thống, feature engineering sẽ tốn nhiều thời gian do kết quả model phụ thuộc rất nhiều vào feature của dữ liệu.

## Tham khảo
- [1] [Giải thích về autocorrelation (tự tương quan)](https://amorfati.xyz/hoc/nhan-dang-va-xu-ly-hien-tuong-tu-tuong-quan-autocorrelation-trong-ols)
- [2] [Giải thích về XGBoost](https://towardsdatascience.com/xgboost-mathematics-explained-58262530904a)
- [3] [Using XGBoost in Python](https://www.datacamp.com/community/tutorials/xgboost-in-python)
- [4] [Basic time series manipulation with pandas](https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea)
- [5] [Tutorial: Time series forecasting with XGBoost](https://www.kaggle.com/robikscube/tutorial-time-series-forecasting-with-xgboost)
