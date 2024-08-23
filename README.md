# Đánh giá thực nghiệm mô hình nhận diện khuôn mặt

## 4.4.1 Chuẩn bị môi trường

- Sử dụng Python 3.10
- Có hai phương pháp để tạo môi trường:
  1. Sử dụng venv của Python
  2. Sử dụng Conda
- Sử dụng pip cài đặt các thư viện cần thiết từ file requirements.txt

## 4.4.2 Chuẩn bị dữ liệu

Cấu trúc thư mục dữ liệu:
root
├── domain1
│ ├── image
│ │ ├── img1.jpg
│ │ ├── img1.txt
│ │ ├── img2.jpg
│ │ ├── img2.txt
│ │ └── ...
│ └── label.txt
├── domain2
│ ├── image
│ │ ├── img1.jpg
│ │ ├── img1.txt
│ │ ├── img2.jpg
│ │ ├── img2.txt
│ │ └── ...
│ └── label.txt
└── ...

- root: Thư mục chứa toàn bộ các thư mục con tương ứng với các domain khác nhau.
- domain: Tên của các domain, đồng thời là các thư mục con trong thư mục root.
- image: Thư mục chứa ảnh và các tệp tọa độ tương ứng, đồng thời là thư mục con trong mỗi thư mục domain.
- label.txt: Tệp định dạng txt chứa thông tin nhãn của ảnh trong thư mục image, cấu trúc mỗi dòng là 'tên_ảnh nhãn'. VD: 'img1.jpg 1'.
- img.jpg: Tệp ảnh định dạng jpg nằm trong thư mục image.
- img.txt: Tệp định dạng txt chứa thông tin tọa độ vùng giới hạn khuôn mặt của ảnh cùng tên trong thư mục image. Tọa độ có cấu trúc 'x y w h'.

Sau khi hoàn tất cấu trúc thư mục dữ liệu, tiến hành sao chép thư mục root này vào thư mục chứa mã nguồn.

## 4.4.3 Huấn luyện mô hình

Thực hiện lệnh `python3 train.py` với các tùy chọn đối số:

| Đối số | Định kiểu | Mặc định | Mô tả |
|--------|----------|----------|-------|
| data_path | str | "" | đường dẫn dữ liệu |
| result_path | str | "./results" | đường dẫn thư mục kết quả |
| batch_size | int | 128 | kích thước lô |
| num_workers | int | 4 | số tiến trình con |
| img_size | int | 256 | kích thước ảnh |
| loo_domain | str | "" | tên domain kiểm định |
| base_lr | float | 5e-4 | tốc độ học khởi tạo |
| num_epochs | int | 200 | số epoch |
| print_freq | int | 20 | tần suất in kết quả kiểm định |
| trans | str | "I" | cách chạy huấn luyện ("I" / "p" / "o") |
| lambda_contrast | float | 0.4 | trọng số hàm PCL |
| lambda_supcon | float | 0.1 | trọng số hàm SupCon |
| momentum | float | 0.9 | quán tính thuật toán tối ưu |
| optimizer | str | "adam" | thuật toán tối ưu ("adam" / "sgd") |
| weight_decay | float | 5e-5 | mức phạt trọng số |
| checkpoint_path | str | "" | đường dẫn điểm lưu |

## Chuyển đổi định dạng mô hình

Thực hiện lệnh `python3 convert_onnx.py` với các tùy chọn đối số:

| Đối số | Định kiểu | Mặc định | Mô tả |
|--------|----------|----------|-------|
| pth_path | str | "" | đường dẫn trọng số mô hình .pth |
| img_size | int | 256 | kích thước ảnh huấn luyện |
| onnx_path | str | "" | đường dẫn mô hình .onnx |
| opset_ver | int | 12 | phiên bản opset onnx |

## Dự đoán sử dụng mô hình ONNX

Sao chép ảnh cần dự đoán vào thư mục images.
Thực hiện lệnh `python3 predict.py` với tùy chọn đối số:

| Đối số | Định kiểu | Mặc định | Mô tả |
|--------|----------|----------|-------|
| onnx_path | str | "" | đường dẫn mô hình .onnx |
| img_name | str | "live.jpg" | tên ảnh dự đoán |
