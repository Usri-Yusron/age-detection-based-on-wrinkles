import cv2

# Fungsi Untuk Mengubah Ukuran Wajah
def resize_face(face, width, height):
    return cv2.resize(face, (width, height), interpolation=cv2.INTER_AREA)

# Fungsi Untuk Mendeteksi Landmark Wajah Menggunakan Sobel Operator
def detect_landmarks(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    h, w = gray.shape
    x = w // 2
    y = h // 2
    landmarks = [
        ((x - 50, y - 7), (50, 15)),  # Mata Kiri
        ((x + 45, y - 7), (50, 15)),  # Mata Kanan
        ((x, y - 90), (120, 55)),     # Dahi
        ((x - 60, y + 25), (35, 15)), # Pipi Kiri
        ((x + 70, y + 25), (35, 15))  # Pipi Kanan
    ]
    return landmarks

# Fungsi Untuk Menerapkan Canny edge Detection Pada Area Sekitar Landmark Dengan Threshold Yang Berbeda
def apply_canny_on_landmarks(face, landmarks, thresholds):
    canny_results = []
    for ((center, (lw, lh)), (low_thresh, high_thresh)) in zip(landmarks, thresholds):
        (lx, ly) = center
        top_left = (lx - lw // 2, ly - lh // 2)
        bottom_right = (lx + lw // 2, ly + lh // 2)
        roi = face[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        edges = cv2.Canny(roi, low_thresh, high_thresh)
        canny_results.append((top_left, edges))
    return canny_results

# Fungsi Untuk Mengelompokkan Usia Berdasarkan Persentase Kerutan
def categorize_age(wrinkle_percentage):
    if wrinkle_percentage < 5:
        return "Muda"
    elif wrinkle_percentage < 15:
        return "Paruh Baya"
    else:
        return "Tua"

# Memuat Klasifier Wajah Yang Telah Dilatih Sebelumnya
klasifier_wajah = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Pastikan Klasifier Berhasil Dimuat
if klasifier_wajah.empty():
    print("Gagal memuat file classifier.")
    exit()

# Membuka Kamera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Gagal membuka kamera.")
    exit()

while True:
    # Membaca Frame Dari Kamera
    ret, frame = video_capture.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Konversi Frame Ke Grayscale
    abu_abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi Wajah Pada Frame
    wajah = klasifier_wajah.detectMultiScale(abu_abu, 1.3, 5)

    # Memeriksa Apakah Ada Wajah Yang Terdeteksi
    for (x, y, w, h) in wajah:
        # Memotong Bagian Wajah
        potongan_wajah = frame[y:y+h, x:x+w]
        lebar_diharapkan = 250
        tinggi_diharapkan = 250
        wajah_diubah = resize_face(potongan_wajah, lebar_diharapkan, tinggi_diharapkan)
        landmarks = detect_landmarks(wajah_diubah)
        thresholds = [
            (0.10, 155),  # Threshold untuk Mata Kiri
            (0.10, 160),  # Threshold untuk Mata Kanan
            (0.08, 170),  # Threshold untuk Dahi
            (0.06, 180),  # Threshold untuk Pipi Kiri
            (0.06, 190)   # Threshold untuk Pipi Kanan
        ]
        canny_landmarks = apply_canny_on_landmarks(wajah_diubah, landmarks, thresholds)

        total_wrinkle_percentage = 0
        for (top_left, edges) in canny_landmarks:
            lx, ly = top_left
            lw, lh = edges.shape[1], edges.shape[0]
            cv2.rectangle(wajah_diubah, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
            wajah_diubah[ly:ly+lh, lx:lx+lw] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            total_piksel_tepi = cv2.countNonZero(edges)
            total_piksel_area = lw * lh
            persentase_tepi = (total_piksel_tepi / total_piksel_area) * 100
            total_wrinkle_percentage += persentase_tepi

        avg_wrinkle_percentage = total_wrinkle_percentage / len(canny_landmarks)
        age_category = categorize_age(avg_wrinkle_percentage)
        cv2.putText(frame, f"Kategori usia: {age_category}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 10, 10), 2)

    # Tampilkan Frame Dengan Wajah Yang Terdeteksi
    cv2.imshow("Deteksi Keriput", frame)

    # Keluar Dari Loop Jika Tombol 'q' Ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup Semua Jendela OpenCV Dan Matikan Kamera
video_capture.release()
cv2.destroyAllWindows()