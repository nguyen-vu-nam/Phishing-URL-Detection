# Dàn ý bảo vệ — Phát hiện URL lừa đảo

## Trình bày và demo

1. Nêu bài toán phân loại nhị phân, đầu vào là URL và lớp cần phát hiện là phishing. Giải thích nhãn 0 trong dữ liệu UCI được ánh xạ thành 1 khi huấn luyện.
2. Giới thiệu 235.795 URL gốc và bước loại 425 dòng lặp. Trình bày cách chia 164.785 URL train, 34.356 validation và 36.229 test với các nhóm tên miền tách biệt.
3. Giải thích 11 đặc trưng từ vựng và TF-IDF n-gram ký tự. Scaler, từ vựng và IDF chỉ khớp trên phần huấn luyện của mỗi lượt.
4. So sánh Logistic Regression, Random Forest, LightGBM và mốc lớp đa số bằng bảng tuần 6. Trình bày thử nghiệm tuần 8 và kết quả cuối bên dưới.
5. Chạy demo với `https://www.example.org` và `https://www.example.org/`. Mô hình cho điểm lần lượt 0,0002 và 0,9613, dù cả hai cùng trỏ đến trang gốc của tên miền ví dụ. Đây là một lỗi có thể tái hiện khi thay đổi hình thức chuỗi URL. Điểm đầu ra được tính tại chỗ. Dùng URL thiếu giao thức để minh họa kiểm tra dữ liệu nhập.
6. Trình bày hạn chế của PhiUSIIL. Trong train, 26,87% URL phishing có đường dẫn khác gốc, so với 0% ở URL hợp lệ. Đánh giá trên nguồn độc lập là bước cần thiết trước khi dùng thực tế.

## Kết quả cuối trên cùng tập test PhiUSIIL

| Cấu hình | Precision | Recall | F1 | ROC-AUC | Bỏ sót | Cảnh báo nhầm |
|---|---:|---:|---:|---:|---:|---:|
| Mốc lớp đa số | 0,00000 | 0,00000 | 0,00000 | 0,50000 | 15.744 | 0 |
| LightGBM tuần 6 | 1,0000 | 0,9959 | 0,9980 | 0,9989 | 64 | 0 |
| LightGBM tuần 8 | 0,99994 | 0,99593 | 0,99793 | 0,99895 | 64 | 1 |

Cấu hình tuần 8 được khóa theo F1 validation và dùng trong demo. F1 validation tăng 0,00004, trong khi kết quả trên cùng test không cải thiện. Việc chỉ lấy n-gram từ máy chủ giảm F1 validation 0,01932 khi giữ nguyên tham số cây.

## Câu hỏi vấn đáp

- **Vì sao chia theo tên miền?** Nếu cùng nhóm xuất hiện ở train và test, điểm có thể phản ánh việc nhận diện tên miền quen thuộc. Chia theo nhóm đo khả năng dự đoán trên tên miền chưa học.
- **Vì sao theo dõi Recall và F1?** Recall đo tỷ lệ phishing phát hiện được. F1 kết hợp Recall và Precision, giúp thấy sự đánh đổi giữa bỏ sót và cảnh báo nhầm.
- **Ba thuật toán khác nhau thế nào?** Logistic Regression học trọng số tuyến tính. Random Forest trung bình dự đoán của các cây bootstrap. LightGBM thêm cây tuần tự theo gradient của hàm mất mát.
- **Vì sao giữ n-gram toàn URL?** Thử nghiệm cùng cách chia cho thấy n-gram máy chủ làm giảm F1. Kết quả này mô tả PhiUSIIL và còn cần kiểm tra trên dữ liệu độc lập.
- **Thực hành tại chỗ:** chạy demo với một đầu vào hợp lệ và một đầu vào sai cú pháp. Trong notebook tuần 8, thay một giá trị trong lưới rồi giải thích cách so sánh bằng CV.

## Hồ sơ nộp

Dùng số liệu và biểu đồ trong bốn notebook để hoàn thiện báo cáo theo mẫu bộ môn. Điền tên, mã sinh viên, phần việc thực tế và đường dẫn commit hoặc nhật ký của từng thành viên. Đối chiếu các nhận xét đã nhận từ giảng viên ở từng mốc trước khi nộp.

[Nguồn dữ liệu UCI](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) · [Bài báo PhiUSIIL](https://doi.org/10.1016/j.cose.2023.103545)

