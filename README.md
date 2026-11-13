# Phát hiện URL lừa đảo (Phishing)

## Phát biểu bài toán

- **Loại bài toán** — Phân loại nhị phân có giám sát.
- **Mục tiêu** — Dự đoán một URL là lừa đảo hay hợp lệ để hỗ trợ phát hiện liên kết phishing.
- **Đơn vị mẫu** — Một URL, tương ứng với một bản ghi trong bộ dữ liệu PhiUSIIL.
- **Đầu vào** — Chuỗi trong cột `URL`, từ đó trích xuất đặc trưng từ vựng gồm độ dài URL, ký tự đặc biệt, tên miền con, dấu hiệu sử dụng địa chỉ IP và từ khóa nhạy cảm, cùng với đặc trưng n-gram ký tự.
- **Đầu ra** — Dự đoán nhãn `label` của UCI. Giá trị `0` là URL lừa đảo và `1` là URL hợp lệ.

## Mục tiêu và tiêu chí đánh giá

Kế hoạch của đề tài là so sánh Logistic Regression, Random Forest và LightGBM. Đánh giá bằng Precision, Recall, F1 và ROC-AUC trên tập kiểm tra gồm các tên miền chưa xuất hiện trong tập huấn luyện. Khi tính các chỉ số, quy ước phishing (nhãn gốc `0`) là lớp dương.

Dùng F1 của lớp phishing làm chỉ số chính. Mức tham chiếu là F1 = 0 của cách luôn dự đoán URL hợp lệ, lớp đa số trong dữ liệu gốc. Mục tiêu tối thiểu là vượt mức tham chiếu này trên tập kiểm tra theo tên miền. Precision và Recall lần lượt phản ánh tỷ lệ cảnh báo đúng và tỷ lệ URL phishing được phát hiện. ROC-AUC đo khả năng xếp URL phishing cao hơn URL hợp lệ theo điểm dự đoán.

## Nguồn và mô tả dữ liệu

Tệp CSV gốc nằm tại `data/PhiUSIIL_Phishing_URL_Dataset.csv`. UCI công bố 235.795 mẫu, gồm 100.945 URL lừa đảo và 134.850 URL hợp lệ. Tệp có 54 đặc trưng thuộc các kiểu phân loại, số nguyên và số thực, cùng cột định danh `FILENAME` và cột nhãn `label`.

Dữ liệu do Arvind Prasad và Shalini Chandra công bố năm 2024 theo giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [PhiUSIIL Phishing URL (Website) — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).

## Bài kiểm tra tuần 4

- [01_eda.ipynb](01_eda.ipynb) trình bày từ điển dữ liệu, thống kê, biểu đồ và các vấn đề chất lượng dữ liệu.
- [02_preprocessing.ipynb](02_preprocessing.ipynb) trình bày cách làm sạch, bảng trước–sau, cách chia theo nhóm tên miền và cách tạo đặc trưng. Mở notebook trong VS Code để xem kết quả đã lưu.
- [data/splits.csv](data/splits.csv) lưu cách chia. Cột `source_row` là chỉ số bản ghi trong CSV gốc, bắt đầu từ 0 sau dòng tiêu đề. Cột `split` cho biết bản ghi thuộc `train`, `validation` hay `test`.

Để chạy lại các notebook, chọn kernel Python 3.12 trong VS Code và cài các thư viện sau vào môi trường của kernel.

```powershell
python -m pip install pandas matplotlib scikit-learn==1.7.2 lightgbm==4.7.0 tldextract==5.4.0 ipykernel
```

Khi vấn đáp, có thể mở notebook EDA, thay một ngưỡng trong phép thống kê hoặc biểu đồ rồi giải thích kết quả. Trong notebook tiền xử lý, kiểm tra bảng số nhóm trùng và giải thích vì sao chỉ khớp các phép biến đổi trên tập huấn luyện.

## Bài kiểm tra tuần 6

Mở [03_training.ipynb](03_training.ipynb) để xem bảng kiểm định chéo, so sánh với mốc lớp đa số, ma trận nhầm lẫn, đường ROC và phân tích lỗi. Notebook dùng lại `data/splits.csv` và khớp TF-IDF riêng trong mỗi lượt kiểm định. Mô hình được chọn theo F1 phishing trên validation trước khi đo trên test.

Trong phần huấn luyện, quy ước nội bộ là 1 = lừa đảo để tính Precision, Recall, F1 và ROC-AUC. Nhãn trong CSV gốc vẫn theo quy ước UCI.

## Bài kiểm tra tuần 8

Mở [04_optimization.ipynb](04_optimization.ipynb) để xem tìm kiếm lưới LightGBM, so sánh từng thay đổi, biểu đồ F1 và thời gian, phân tích nhóm dễ bỏ sót và đánh giá cuối. Notebook dùng cùng `data/splits.csv` và cùng ba lượt kiểm định theo tên miền của tuần 6.

Cấu hình được chọn bằng F1 trên validation sau khi tìm tham số qua kiểm định chéo trên train. Tập test PhiUSIIL từng dùng để báo cáo ở tuần 6 và được dùng lại để so sánh sau khi khóa cấu hình tuần 8.

## Bài thi kết thúc — Tuần 9

Cài thư viện trong môi trường Python 3.12 và chạy demo từ thư mục dự án.

```powershell
python -m pip install -r requirements.txt
python demo.py --url "https://www.example.org"
```

Demo dùng mô hình LightGBM tuần 8 trong `models/lightgbm.joblib`. Để tạo lại từ tập train đã lưu, chạy `python demo.py --train`. Chỉ nạp tệp mô hình đi kèm dự án hoặc tệp do chính bạn tạo lại. Chương trình phân tích chuỗi tại chỗ và xuất nhãn cùng điểm phishing, với ngưỡng 0,5.

Xem [BAO_VE.md](BAO_VE.md) để chuẩn bị phần trình diễn, giải thích kết quả và vấn đáp.


## Chạy ứng dụng Windows

Sau khi giải nén và cài thư viện, bấm đúp `Mo-App.cmd` hoặc chạy `py -3.12 app.py` trong thư mục dự án. Dán URL đầy đủ rồi bấm **Kiểm tra URL** hoặc nhấn Enter. Nhấn Esc để xóa. Ứng dụng dùng cùng mô hình với demo dòng lệnh và xử lý chuỗi URL tại chỗ.

App nhận diện riêng các trang chủ HTTPS đã đối chiếu, có thể có một dấu / ở cuối. Đường dẫn con, tham số, cổng và tên miền khác vẫn được đưa vào LightGBM.

- [ChatGPT theo tài liệu OpenAI](https://help.openai.com/en/articles/12677804-what-is-chatgpt-faq)
- [Trang chủ Google Dịch](https://translate.google.com/)
- [Trang chủ GitHub](https://github.com/)

Với URL ngoài danh sách trên, app hiện **URL chưa được xác minh** cùng tín hiệu LightGBM cao hoặc thấp. Điểm mô hình không xác nhận tính hợp lệ hay tình trạng lừa đảo của một liên kết.

## Đánh giá ngoài PhiUSIIL

Xem [kết quả huấn luyện lại và kiểm tra ngoài](DANH_GIA_DU_LIEU_NGOAI.md). Bản thử được lưu riêng để đối chiếu.


## Kiểm tra hàng loạt bằng link.txt

1. Mở `link.txt`, mỗi dòng nhập một URL đầy đủ bắt đầu bằng `http://` hoặc `https://`.
2. Lưu file.
3. Chạy `python app.py`.
4. Kết quả hiển thị trong cửa sổ PowerShell và được lưu vào `ket_qua.txt`.

Dòng trống và dòng bắt đầu bằng `#` được bỏ qua. `app.py` không mở URL và không truy cập website; nó chỉ phân tích chuỗi URL bằng model hiện có.
