# Quét độc lập trước khi xem pre-label

`Frame: frame_0182.jpg`

Số xe nhìn thấy bằng mắt: 25

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe: 

1. Ảnh ở góc trên cùng bên trái, ảnh thiếu thông tin do đã đến điểm nhìn giới hạn của camera
2. Ảnh ở giữa khung hình bên dưới, xe đã sắp vượt qua tầm nhìn của camera nên có thể bị nhận nhầm sang lớp khác nếu có

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
