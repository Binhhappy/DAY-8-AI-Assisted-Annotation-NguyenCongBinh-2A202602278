# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Công Bình

Công cụ gán nhãn đã dùng: CVAT chạy Docker local (task riêng `day8_round1`, import/export định dạng Ultralytics YOLO Detection 1.0)

Nguồn số liệu: `reports/rounds_table.md`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`,
`outputs/selection_round1.csv`, `outputs/selection_round2.csv`, `outputs/round1_diff.md`, `outputs/compare_round0.jpg`,
`outputs/compare_round1.jpg`. Nhãn test do một mô hình tạo, chưa được người rà; mọi số đo dưới đây là mức khớp với bộ
tham chiếu đó, không phải chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Camera đặt cố định trên cầu vượt, video lấy mẫu 2.5 frame/giây, nên hai frame liền nhau (cách 0.4 s) gần như giống hệt
và mỗi chiếc xe ở lại trong khung hình vài giây. Nếu chia pool/test ngẫu nhiên, cùng một chiếc xe ở cùng vị trí gần như
chắc chắn xuất hiện cả trong ảnh train lẫn ảnh test. Mô hình khi đó được chấm trên những chiếc xe nó đã "thấy", nên số đo
trên test sẽ **lệch lên (lạc quan hơn thực tế)**. Đây là rò rỉ dữ liệu (data leakage).

Repo chia theo trục thời gian: 20 ảnh test ở 4 đoạn có tâm 20/60/100/140 s, bỏ 112 ảnh vùng đệm ±4 s quanh mỗi đoạn,
268 ảnh còn lại là pool (`data/DATA.md`). Ảnh pool gần test nhất vẫn cách 4.4 s. Tôi không sửa `data/test/labels/`
và không đưa ảnh test vào lô gán nhãn (`pack_labels.py` kiểm và chặn tên ảnh test).

## 2. Mô hình khởi đầu lạnh (cold start)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Cold start là YOLOv8n COCO, gộp lớp car/bus/truck thành `car`. Trên 20 ảnh test (403 box tham chiếu tính điểm, bỏ 14 box
cao dưới 16 px) nó có TP 197, FP 16, FN 206 (`metrics_round0.json`). Precision cao (0.925) nhưng recall chỉ 0.489:
**lỗi chính là bỏ sót, không phải báo nhầm.**

Theo `compare_round0.jpg`:

- **Xe nhỏ ở xa gần đường chân trời** bị bỏ sót nhiều nhất (recall small 0.182 trên 66 box). Ở cả bốn ảnh, các box vàng (FN)
  dày đặc ở dãy xe phía xa bên trái, nơi chỉ còn cụm đèn.
- **Xe lớn gần camera, bị lóa đèn pha** cũng bị sót dù to: frame_0050 (xe góc dưới trái) và frame_0250 (xe góc dưới trái,
  sát mép dưới) đều là box vàng. Recall medium 0.547 và large 0.561 gần bằng nhau, nghĩa là kích thước lớn không giúp
  model khi xe bị ánh đèn phủ.
- **FP** chủ yếu là box gộp nhiều xe: frame_0350 có một box đỏ lớn phủ cụm xe làn trái; frame_0150 có các box chồng nhau ở
  cụm đèn hậu bên phải.

Ca cần người rà lại nhãn tham chiếu trước khi kết luận model sai: **frame_0150, cụm đèn hậu đỏ bên phải**. Bộ tham chiếu
có ba box chồng lên nhau ở đó, trong khi mặt đường quanh cụm này bị ánh đỏ phủ. Có thể tham chiếu đã vẽ đúng ba xe đi
sát nhau, nhưng cũng có thể một box ôm vệt sáng. Một ca tương tự là **box tham chiếu rất mảnh sát mép phải frame_0250**,
trông giống vệt sáng hơn là thân xe. Cả hai model đều "bỏ sót" box này; nếu đó không phải xe thì đây không phải lỗi của model.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D` với trọng số 0.5 / 0.3 / 0.2 (`tools/al_select.py`):

- **U (bất định)**: với mỗi box dự đoán, `u = 1 − |2c − 1|`, bằng 1 khi confidence c = 0.5 (model phân vân nhất). U là trung
  bình 5 giá trị u lớn nhất trong frame, tức độ phân vân ở những box khó nhất.
- **A (mập mờ)**: số box có 0.15 ≤ c < 0.50, chia cho giá trị lớn nhất trong pool. Frame có nhiều xe "lưng chừng" thì A cao.
- **D (đa dạng)**: khoảng cách thời gian tới frame đã gán gần nhất, chặn ở 10 s. Vòng 1 chưa có frame nào đã gán nên D = 1 cho
  mọi frame.
- **MIN_GAP_S = 2.0**: khi chọn tham lam theo score, bỏ frame nằm cách frame đã chọn dưới 2 s. Camera đứng yên, hai frame sát nhau
  gần như trùng cảnh, gán cả hai tốn công mà model học thêm rất ít.

Theo `reports/SELECTION.md`:

- **frame_0182** (rank 1, score 0.9591, A = 1.0, 18 box mập mờ): pre-label 13 box, sau khi rà 26 box (added 13).
- **frame_0369** (rank 2, score 0.9324, 43 box dự đoán): phải thêm nhiều nhất lô, 14 → 37 box (added 24).
- **frame_0392** (rank 15, U = 0.9747 cao nhất top 50): chỉ vào được lô vì frame_0372, 0368 và 0330 bị loại do gần trùng.
- Frame cân nhắc khác: **frame_0330** (rank 12, 53 box, đông nhất top 50) bị loại vì cách frame_0331 chỉ 0.4 s. Đây vừa là
  ảnh gần trùng vừa là ảnh tốn công rà nhất.

Điểm bất định **không** chứng minh ảnh đó sẽ cải thiện model. U và A đo mức phân vân của chính model cold start, không đo lỗi
thật hay lượng thông tin model sẽ học được. Vòng này là ví dụ trực tiếp: lô 12 ảnh có score cao nhất vẫn cho model sau train
kém hơn cold start (mục 4). Không có lô đối chứng `random` nên cũng không so được với chọn ngẫu nhiên.

## 4. Các vòng học chủ động (active learning)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 338 | 0.517 | -0.254 | 1.000 | 0.070 | 0.130 | 0.000 | 0.068 | 0.195 |

**Mức sửa pre-label vòng 1** (`round1_diff.md`): model đề xuất 169 box (conf ≥ 0.25), nhãn cuối có 338 box.
Accepted 162, edited 2, deleted 5, added 174, accept rate 96%. Pre-label hiếm khi sai vị trí, nhưng **thiếu gần một nửa số
xe**: cứ mỗi box AI đề xuất, tôi phải thêm khoảng một box nữa.

**Kết quả trên cùng 20 ảnh test:** AP50 giảm 0.254 (0.771 → 0.517). Recall@0.25 sụp từ 0.489 xuống 0.070 (TP 197 → 28,
FN 206 → 375). Precision tăng lên 1.000 (FP 16 → 0). Mọi nhóm kích thước đều xấu đi: small 0.182 → 0.000, medium
0.547 → 0.068, large 0.561 → 0.195. Mức giảm này lớn hơn nhiều ngưỡng nhiễu ~0.01 AP50 của tập test 20 ảnh.

**Ca đổi sau fine-tune** (`compare_round1.jpg`, frame_0350): cold start có 9 TP, 2 FP, 14 FN, gồm cả box đỏ lớn gộp cụm xe
làn trái. Model vòng 1 còn 2 TP, 0 FP, 21 FN. Box gộp sai đã biến mất, nhưng gần như mọi xe khác cũng không còn box trên ngưỡng
0.25; hai TP còn lại là hai xe lớn gần camera nhất. Ba ảnh test còn lại cho cùng kiểu: TP 11/10/6 → 0/2/2.

**Vì sao tôi cho rằng nguyên nhân chính là cách fine-tune, không phải nhãn sai** (các điểm dưới đây kiểm được):

1. AP50 tính trên mọi dự đoán từ conf 0.01, còn P/R tính ở conf 0.25. AP50 còn 0.517 trong khi recall@0.25 chỉ 0.070, nghĩa là
   model vẫn xếp hạng đúng nhiều xe nhưng với **confidence rất thấp**, dưới 0.25.
2. Trên pool, ở conf ≥ 0.05 (cột n_boxes), model vòng 1 chỉ dự đoán trung vị **5 box/frame** (1–10, `selection_round2.csv`),
   trong khi cold start dự đoán trung vị **28 box/frame** (14–53, `selection_round1.csv`). Pre-label vòng 2 chỉ còn 29 box cho 12 ảnh, so với 169 box ở vòng 1.
3. Notebook train từ `yolov8n.pt` với `names: {0: car}`. Đầu phân lớp COCO 80 lớp bị thay bằng đầu 1 lớp khởi tạo mới, rồi chỉ
   được học từ 12 ảnh × 50 epoch, batch 16, tức khoảng 50 bước cập nhật. Theo tôi, mất lợi thế lớp car/bus/truck của COCO trong
   khi đầu mới chưa hội tụ là giải thích hợp lý nhất cho confidence thấp đồng loạt.
4. Nhãn train khớp guideline qua các ca tôi kiểm lại bằng mắt (`REVIEW_LOG.csv`), và số box nhãn (338) nhiều gấp đôi pre-label.
   Nếu nhãn sai theo kiểu vẽ thừa, model sẽ có thêm FP, nhưng FP lại về 0.

Đây là giả thuyết, chưa phải kết luận. Để khẳng định, cần chạy model vòng 1 trên chính 12 ảnh train và xem confidence (mục 5).

**Phân biệt ba nguồn bằng chứng:**

- **Quan sát độc lập** (`BLIND_SCAN.md`, frame_0182): trước khi xem pre-label tôi đếm 25 xe và đoán AI sẽ sót ở góc trên trái
  (xe ở xa) và giữa mép dưới (xe sắp ra khỏi khung hình). Pre-label có 13 box. Nhãn cuối có 26 box, và các box tôi thêm đúng
  gồm 4 xe nhỏ đèn đỏ ở góc trên trái và xe tối bị cắt ở giữa mép dưới.
- **Lỗi pre-label đã sửa** (`REVIEW_LOG.csv`, `round1_diff.md`): chủ yếu là bỏ sót (174 added), như xe lớn đèn pha trắng giữa
  mép dưới frame_0099. Ngoài ra có box gộp hai xe (frame_0326, deleted), box chỉ ôm cụm đèn hậu (frame_0187, deleted), box phủ
  nửa xe tải (frame_0392, edited).
- **Kết quả model sau train** (`metrics_round1.json`): kém hơn cold start. Nhãn tốt hơn pre-label không tự đảm bảo model tốt hơn
  khi quy trình train chưa phù hợp.

**Ca khó theo guideline và cách tôi giữ nhất quán:** frame_0331 có một xe tối bị nhòe chuyển động sát mép dưới phải. AI chỉ khoanh
vệt phản sáng trên thân (x 1023–1088). Theo luật "xe nhòe vẫn gán, box ôm vùng nhòe của thân xe" và "xe cắt mép chỉ vẽ phần trong
ảnh", tôi vẽ lại box ôm toàn bộ thân nhòe tới mép ảnh (x 1023–1246). IoU với box cũ dưới 0.5 nên `round1_diff` tính là 1 deleted
+ 1 added. Quy ước tôi dùng cho cả lô: không tính vệt sáng đèn pha trên mặt đường; xe cắt mép ảnh vẽ tới sát mép; hai xe đi sát nhau
luôn tách hai box; xe ở xa chỉ vẽ khi tách được cặp đèn của từng xe.

**Ghi chú tự khai về blind scan:** nội dung `BLIND_SCAN.md` hoàn tất lúc 11:26 (giờ máy). File được khóa lại lần cuối lúc 12:31, vì
các lần khóa trước bị lệch hash do tôi chỉnh định dạng sau khi khóa. Lần import pre-label đầu tiên vào CVAT (12:26) nạp 0 box do lỗi
đường dẫn trong file ZIP, nên trước khi khóa tôi chưa thấy box AI nào. Tôi chỉ xem pre-label sau khi import lại thành công.

## 5. Kết luận và giới hạn

**So với cold start**, model vòng 1 kém rõ rệt: AP50 0.517 so với 0.771, recall@0.25 0.070 so với 0.489. Precision 1.000 chỉ vì
model gần như không dám dự đoán. **Tôi dừng, không làm vòng 2 với cấu hình hiện tại.** Lô vòng 2 đã được chọn
(`selection_round2.csv`) nhưng pre-label chỉ có 29 box cho khoảng 12 × 25 xe, nên gán vòng 2 gần như là vẽ tay từ đầu (khoảng 300
box). Nếu train tiếp cùng cách, nhiều khả năng lại mất độ tự tin như vòng này.

**Hai ca còn yếu hoặc bất định cho vòng sau:**

1. **Xe nhỏ ở xa gần đường chân trời**: recall small 0.182 ở cold start, 0.000 sau vòng 1. Ví dụ dãy xe phía xa bên trái
   frame_0350. Chi phí rà cao, vì mỗi ảnh có hàng chục cặp đèn nhỏ, phải zoom và quyết định ngưỡng 16 px. Guideline cho phép bỏ
   box dưới 16 px, nên chỉ nên tập trung vào xe từ 16 px trở lên.
2. **Xe lớn gần camera bị lóa đèn pha**: frame_0050 và frame_0250 (xe góc dưới trái) bị cả hai model bỏ sót. Nên chọn thêm ảnh
   có xe gần đi tới với đèn pha mạnh. Chi phí rà thấp vì ít xe và xe to, nhưng cần rà lại box tham chiếu trước, vì có thể lệch IoU
   do vệt sáng.

**Nguy cơ ảnh gần trùng:** vòng 2 chọn frame_0376 và frame_0387 dù D chỉ 0.16 và 0.20, tức nằm rất gần các frame vòng 1
(0369/0380/0392). Nhóm frame_0072–0075 (rank 2, 3, 5, 6) cũng gần như cùng một cảnh. Vì U của model vòng 1 thấp đồng loạt, score
vòng 2 (0.63–0.66) phân biệt kém, nên lô vòng 2 dễ gồm nhiều cảnh gần giống nhau.

**Giới hạn ảnh hưởng tới kết luận:**

- Tập test chỉ 20 ảnh ở 4 đoạn thời gian. Chênh dưới khoảng 0.01 AP50 không có ý nghĩa. Mức giảm 0.254 ở đây đủ lớn để kết luận
  model kém đi, nhưng không đủ để nói kém ở mọi điều kiện.
- 14 box tham chiếu cao dưới 16 px bị bỏ qua. Vì vậy recall small chỉ phản ánh xe nhỏ từ 16 px trở lên, và việc tôi vẽ hay không
  vẽ xe cực nhỏ không ảnh hưởng tới số đo.
- Nhãn tham chiếu do mô hình tạo, chưa được người rà. Kiểu xe mà model tham chiếu cũng hay sót (xe rất tối, xe bị che) có thể
  vắng trong tham chiếu, nên số đo có thể phạt nhầm model khi nó đúng, hoặc thưởng model khi nó mắc cùng lỗi với tham chiếu.
  Các ca ở mục 2 (frame_0150 và frame_0250) cần người rà trước khi tính là lỗi.

**Trước khi train thêm, tôi sẽ kiểm theo thứ tự:**

1. Chạy model vòng 1 trên chính 12 ảnh train. Nếu recall trên train cũng thấp, lỗi nằm ở quá trình train (chưa hội tụ), không
   phải ở nhãn hay độ khác biệt pool/test.
2. Xem phân bố confidence trên test, và tính recall ở ngưỡng thấp hơn (ví dụ 0.05) để xác nhận giả thuyết "đúng nhưng thiếu tự tin".
3. Đối chiếu ngẫu nhiên vài file `labels/round1/*.txt` với ảnh, để loại trừ lỗi định dạng hoặc lệch tọa độ khi export từ CVAT.
4. Sau đó mới thử đổi cách train, mỗi lần một yếu tố: nhiều epoch hơn, đóng băng backbone, hoặc giữ đầu COCO rồi ánh xạ
   car/bus/truck thành `car` thay vì khởi tạo đầu 1 lớp mới. Nên chạy thêm một lô `random` cùng kích thước làm đối chứng.
   Không sửa nhãn test hay số đo để kéo AP50 lên.
