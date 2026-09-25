# Vì sao chọn lô này?

Nguồn: `outputs/selection_round1.csv` (268 frame pool, chiến lược `uncertainty`, `score = 0.5·U + 0.3·A + 0.2·D`,
`MIN_GAP_S = 2.0`) và contact sheet `outputs/selection_round1.jpg`. Ở vòng 1 chưa có frame nào đã gán nên
`D = 1` cho mọi frame; thứ hạng chỉ do U (độ bất định của 5 box khó nhất) và A (tỉ lệ box mập mờ 0.15–0.50) quyết định.

## Top 5 nếu chỉ đủ công rà năm ảnh

Trong 50 dòng đầu, score chỉ dao động 0.820–0.959 và cả 50 frame đều có 24–53 box dự đoán, không frame nào
trống (`empty = False` cho toàn pool). Vì chênh lệch score nhỏ, tôi ưu tiên phủ nhiều đoạn thời gian khác nhau
thay vì lấy đúng 5 dòng đầu:

| thứ tự | frame | rank | score | t (s) | U | A | n_boxes | lý do |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 0.9591 | 72.8 | 0.918 | 1.000 | 28 | Score cao nhất, A tối đa (18 box mập mờ): model phân vân nhiều nhất |
| 2 | frame_0369.jpg | 2 | 0.9324 | 147.6 | 0.932 | 0.889 | 43 | Cảnh đông xe cuối video, U cao |
| 3 | frame_0326.jpg | 4 | 0.9155 | 130.4 | 0.931 | 0.833 | 39 | Đoạn 124–133 s có nhiều frame điểm cao (0312, 0326, 0330, 0331), lấy một đại diện |
| 4 | frame_0099.jpg | 8 | 0.9063 | 39.6 | 0.946 | 0.778 | 29 | Đoạn đầu video chưa được đại diện; U cao nhất trong top 10 |
| 5 | frame_0227.jpg | 11 | 0.8915 | 90.8 | 0.916 | 0.778 | 37 | Phủ đoạn giữa 72–130 s |

Quyết định xét ảnh gần trùng: tôi **không** lấy frame_0380 (rank 3, 0.917) và frame_0331 (rank 5, 0.9154)
dù score cao hơn frame_0099 và frame_0227. frame_0380 cách frame_0369 chỉ 4.4 s, còn frame_0331 cách
frame_0326 đúng 2.0 s (vừa chạm `MIN_GAP_S`). Camera đứng yên và mỗi xe ở trong khung hình vài giây, nên hai
frame gần nhau chứa phần lớn cùng những chiếc xe. Với ngân sách 5 ảnh, rà thêm một cảnh gần trùng tốn công
(frame_0331 có 47 box) mà model học thêm ít hơn so với một đoạn thời gian mới. Năm frame đã chọn nằm ở 39.6,
72.8, 90.8, 130.4 và 147.6 s, cách nhau ít nhất 17 s.

Pool không có frame nào model dự đoán 0 box, nên luật `EMPTY_BONUS` không được kích hoạt ở vòng này.

## Ba frame thuộc lô 12 ảnh model chọn và bằng chứng

- **frame_0182.jpg**: rank 1, score 0.9591, U 0.918, A 1.0, 28 box, 18 box mập mờ. Pre-label (conf ≥ 0.25) chỉ
  giữ 13 box; sau khi rà tôi có 26 box (`round1_diff.md`: accepted 13, added 13). Điểm bất định cao đúng là
  dấu hiệu model đang để nhiều xe ở dưới ngưỡng 0.25. Trên contact sheet, đây là cảnh có xe lớn sát mép dưới
  và dãy xe nhỏ phía xa.
- **frame_0369.jpg**: rank 2, score 0.9324, U 0.932, 43 box dự đoán, 16 box mập mờ. Đây là frame phải thêm
  nhiều box nhất (pre-label 14 box → 37 box sau khi sửa, added 24). Contact sheet cho thấy mật độ xe rất cao,
  nhiều xe đi sát nhau và bị che.
- **frame_0392.jpg**: rank 15, score 0.8874, nhưng có **U cao nhất trong top 50 (0.9747)**; A thấp hơn (0.667,
  12 box mập mờ). Frame này được vào lô vì ba frame rank 6, 9, 12 bị loại do gần trùng (xem dưới). Khi rà, đây là
  frame có 2 box `edited` duy nhất của lô: một xe tải dài bị box phủ nửa thân và một xe bị cắt ở mép phải.

## Một frame điểm cao nhưng không được chọn

**frame_0372.jpg** (rank 6, score 0.9101, U 0.920) và **frame_0368.jpg** (rank 9, 0.9003) không được chọn vì cách
frame_0369 đã chọn lần lượt 1.2 s và 0.4 s, dưới `MIN_GAP_S = 2.0`. **frame_0330.jpg** (rank 12, 0.8899) là frame
đông nhất top 50 với **53 box**, nhưng chỉ cách frame_0331 0.4 s. Bỏ các frame này là hợp lý: chúng gần như trùng
cảnh đã chọn, và frame_0330 còn là ảnh tốn công rà nhất. Nếu có vòng 2 với ngân sách lớn hơn, frame_0330 đáng xem
lại để kiểm tra model sau fine-tune còn phân vân ở cảnh rất đông hay không, vì lúc đó `D` sẽ phạt nó do nằm gần
frame đã gán.

## Điều phép chọn này chưa chứng minh về chất lượng mô hình

- U và A đo **mức phân vân của chính model cold start**, không đo lỗi thật. Một frame bất định cao chưa chắc giúp
  model tốt lên sau khi train; điều đó chỉ kiểm được bằng `metrics_round1.json` trên tập test.
- Không có lô đối chứng `strategy="random"` với cùng số ảnh, nên không thể nói chọn theo bất định tốt hơn chọn
  ngẫu nhiên.
- Score của top 50 rất sát nhau (0.82–0.96), nên thứ tự trong nhóm này nhạy với nhiễu confidence của model.
- Ngưỡng pre-label 0.25 làm ẩn các box mập mờ; số box người thêm (174 trên 12 ảnh) cho thấy recall pre-label
  thấp, nhưng đó là đánh giá nhãn gợi ý, không phải đánh giá model sau fine-tune.
