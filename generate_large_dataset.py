"""
SCRIPT TỰ ĐỘNG SINH 3000 DÒNG DATASET PHÂN LOẠI Y TẾ
--------------------------------------------------
Sử dụng phương pháp Template-based Data Augmentation giúp sinh dữ liệu
nhanh chóng, đa dạng và cân bằng mà không tốn phí API.
"""

import random
import csv

# --- 1. DANH SÁCH TỪ VỰNG Y TẾ ---
symptoms = [
    "đau đầu", "sốt cao", "ho khan", "tê bì chân tay", "ù tai", "đau dạ dày", "mẩn đỏ", 
    "ngứa dị ứng", "chóng mặt", "buồn nôn", "đau bụng lâm râm", "khản tiếng", "mất ngủ", 
    "đau nhức xương khớp", "trào ngược dạ dày", "sốt co giật", "đau khớp gối", "đau thắt lưng",
    "viêm họng hạt", "khó thở", "tức ngực", "mệt mỏi", "rụng tóc nhiều", "chảy máu chân răng",
    "đau răng buốt", "nổi hạch ở cổ", "đau vai gáy", "đi ngoài phân lỏng", "co thắt đại tràng",
    "khô mắt", "giảm thị lực", "tiểu buốt", "tiểu đêm nhiều lần", "đau vùng hạ vị", "nổi mề đay",
    "đau khớp cổ tay", "đau gót chân", "ợ chua liên tục", "khó nuốt nước bọt", "nổi mụn nước"
]

diseases = [
    "sốt xuất huyết", "tiểu đường tuýp 2", "suy thận cấp", "gút (gout)", "viêm gan B", "sỏi thận",
    "zona thần kinh", "loãng xương", "thoái hóa cột sống", "trầm cảm", "viêm phổi", "hen suyễn",
    "viêm xoang mãn tính", "đột quỵ", "nhồi máu cơ tim", "viêm tai giữa", "viêm loét dạ dày",
    "cúm A", "covid-19", "thủy đậu", "quai bị", "viêm phế quản", "viêm amidan", "trĩ nội",
    "rối loạn tiền đình", "gan nhiễm mỡ", "huyết áp cao", "suy nhược cơ thể", "viêm khớp dạng thấp"
]

doctors = [
    "Nguyễn Văn A", "Trần Thị B", "Lê Văn C", "Phạm Minh D", "Hoàng Kim E", "Vũ Thị F", 
    "Đỗ Hoàng G", "Bùi Văn H", "Ngô Quốc I", "Phan Thanh K", "Lý Thu L", "Nguyễn Tiến M",
    "Phạm Đức N", "Lê Thị O", "Hoàng Văn P", "Trần Quốc Q", "Nguyễn Minh R", "Đặng Văn S"
]

specialties = [
    "cơ xương khớp", "tim mạch", "da liễu", "tiêu hóa", "thần kinh", "nhi khoa", "tai mũi họng",
    "phụ khoa", "nam khoa", "nội tiết", "hô hấp", "mắt", "nha khoa", "chấn thương chỉnh hình",
    "sản phụ khoa", "thận tiết niệu", "tâm thần", "ung bướu", "nội tổng quát", "phục hồi chức năng"
]

provinces = ["Hà Nội", "TP.HCM", "Đà Nẵng", "Cần Thơ", "Huế", "Nha Trang", "Biên Hòa", "Hải Phòng", "Vinh", "Quảng Ninh", "Bình Dương", "Vũng Tàu"]
hospitals = ["Bạch Mai", "Chợ Rẫy", "Đại học Y Dược", "Từ Dũ", "108", "Việt Đức", "Vinmec", "Medlatec", "Hồng Ngọc", "Thu Cúc", "Hoàn Mỹ", "FV"]
procedures = ["chụp X-quang phổi", "chụp CT Scanner đầu", "nội soi dạ dày gây mê", "xét nghiệm máu", "siêu âm thai", "xét nghiệm ADN", "xét nghiệm đường huyết", "đo điện tâm đồ", "chụp MRI cột sống"]
days = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]
times = ["sáng", "chiều", "tối", "lúc 9h sáng", "lúc 14h chiều", "khung giờ trống", "lúc 8h30", "lúc 15h45"]

# --- 2. DANH SÁCH TỪ VỰNG NGOÀI LỀ (NGOÀI Y TẾ) ---
subjects = ["Tôi", "Bạn em", "Anh trai em", "Chị gái", "Đồng nghiệp", "Khách hàng", "Người mới học", "Lập trình viên"]
tech_topics = ["ReactJS", "Node.js", "Express API", "Python Flask", "SQL Server", "MongoDB", "TailwindCSS", "Redux Toolkit", "Next.js", "GitHub Action", "Docker", "AWS Cloud", "Linux OS"]
tech_verbs = ["sửa lỗi kết nối database", "tối ưu hiệu năng ứng dụng", "triển khai deploy dự án", "cấu hình phân quyền", "viết hàm gọi API", "cài đặt thư viện", "xử lý bất đồng bộ", "thiết kế giao diện", "quản lý state"]
cooking_verbs = ["nấu canh chua", "kho cá lóc", "làm bánh mì", "pha trà sữa", "làm kim chi cải thảo", "xào sườn chua ngọt", "nấu phở bò", "nướng thịt heo", "ủ sữa chua", "làm salad giảm cân"]
locations = ["Đà Lạt", "Phú Quốc", "Nha Trang", "Sapa", "Hạ Long", "Hội An", "Bản Giốc", "Mộc Châu", "Đà Nẵng", "Vũng Tàu", "Mũi Né", "Tràng An", "Yên Tử"]
songs = ["Lạc Trôi", "Em Của Ngày Hôm Qua", "Nơi Này Có Anh", "Chúng Ta Của Hiện Tại", "Chạy Ngay Đi", "Ngày Mai Người Ta Lấy Chồng", "Bao Tiền Một Mớ Bình Yên"]
singers = ["Sơn Tùng M-TP", "Đen Vâu", "Mỹ Tâm", "Soobin Hoàng Sơn", "Hoàng Thùy Linh", "Hà Anh Tuấn", "Phan Mạnh Quỳnh", "Trúc Nhân"]
games = ["Liên Quân Mobile", "GTA 5", "PUBG Mobile", "Minecraft", "FIFA Online 4", "Liên Minh Huyền Thoại", "Valorant", "Genshin Impact"]
universities = ["Bách Khoa Hà Nội", "Kinh tế Quốc dân", "Ngoại thương", "Công nghệ Thông tin", "Y Dược TP.HCM", "Khoa học Tự nhiên", "Tôn Đức Thắng"]

# --- 3. ĐỊNH NGHĨA TEMPLATE ---
medical_templates = [
    "Tôi bị {symptom} và {symptom} thì nên uống thuốc gì?",
    "Lịch khám của bác sĩ {doctor} vào {day} thế nào?",
    "Triệu chứng của bệnh {disease} là gì?",
    "Cho tôi hỏi giá khám chuyên khoa {specialty} tại bệnh viện {hospital}",
    "Tôi muốn đặt lịch khám {specialty} vào {day} {time}",
    "Dấu hiệu nhận biết sớm của bệnh {disease} ở người già là gì?",
    "Trẻ em bị {symptom} kéo dài có nguy hiểm không bác sĩ?",
    "Cho tôi xem danh sách các phòng khám {specialty} tốt ở {province}",
    "Làm thế nào để giảm cân an toàn khi đang điều trị {disease}?",
    "Bác sĩ {doctor} chuyên khoa {specialty} khám ở đâu tại {province}?",
    "Triệu chứng của bệnh {disease} cấp tính là gì?",
    "Giá tiền một ca {procedure} là bao nhiêu?",
    "Bị đau nhức ở vùng {symptom} khi trời lạnh phải làm sao?",
    "Tôi bị {symptom} kéo dài hơn một tháng nay thì khám ở khoa nào?",
    "Bác sĩ chuyên khoa {specialty} nào khám ngoài giờ ở {province}?",
    "Dấu hiệu của bệnh {disease} giai đoạn đầu là gì?",
    "Tôi muốn đặt lịch tư vấn {specialty} trực tuyến",
    "Bệnh {disease} có chữa dứt điểm được không?",
    "Bị nổi {symptom} nên bôi thuốc gì để nhanh khỏi?",
    "Phòng khám {specialty} nào làm việc ngày {day}?",
    "Bác sĩ chuyên khoa {specialty} có khám vào {day} không?",
    "Các bài tập thể dục giúp giảm {symptom} hiệu quả",
    "Triệu chứng {disease} xuất hiện đột ngột như thế nào?",
    "Có cần nhịn ăn trước khi đi {procedure} không?",
    "Bác sĩ {doctor} chuyên khoa {specialty} có phòng khám riêng không?",
    "Bị {symptom} uống thuốc gì nhanh khỏi?",
    "Phòng khám {specialty} của bác sĩ {doctor} địa chỉ ở đâu?",
    "Đặt lịch khám {specialty} định kỳ cho bé ở đâu tốt?",
    "Bác sĩ chuyên khoa {specialty} tốt nhất ở bệnh viện {hospital}",
    "Thời gian ủ bệnh của bệnh {disease} là bao lâu?",
    "Xét nghiệm {procedure} thực hiện vào thời gian nào?",
    "Chi phí {procedure} bao nhiêu tiền?",
    "Đăng ký khám với bác sĩ chuyên khoa {specialty} {doctor}",
    "Đặt lịch khám {specialty} trị {symptom} vào {time} {day}"
]

non_medical_templates = [
    "Chào chatbot, thời tiết ở {province} hôm nay thế nào?",
    "Trận bóng đá tối nay mấy giờ chiếu thế?",
    "Làm thế nào để học giỏi {tech_topic}?",
    "Cách {cooking_verb} ngon tại nhà chuẩn vị như thế nào?",
    "Thủ đô của nước {province} tên là gì?",
    "Làm sao để {tech_verb} trong dự án {tech_topic}?",
    "Công thức {cooking_verb} cực kỳ chi tiết tại nhà",
    "Làm thế nào để {tech_verb} nhanh nhất?",
    "Hôm nay có tin tức gì mới về tình hình kinh tế thế giới không?",
    "Làm thế nào để viết một bài luận tiếng Anh đạt điểm cao?",
    "Mua vé máy bay đi du lịch {location} ở đâu rẻ nhất?",
    "Cách khắc phục khi điện thoại bị treo màn hình",
    "Kể tên các bài hát hay nhất của ca sĩ {singer}",
    "Cách trồng và chăm sóc hoa hồng trên ban công chung cư",
    "Tải game {game} bản mới nhất ở trang web nào uy tín?",
    "Làm sao để kiếm tiền online cho sinh viên thời nay?",
    "Hướng dẫn cắm hoa cúc đẹp để bàn thờ ngày Tết",
    "Những bộ phim chiếu rạp nào đang hot và đáng xem tuần này?",
    "Làm thế nào để đăng ký tài khoản mới trên mạng xã hội?",
    "Giá vàng hôm nay tăng hay giảm so với hôm qua?",
    "Cách viết một CV xin việc ấn tượng cho sinh viên mới ra trường",
    "Làm sao để bảo mật tài khoản cá nhân trực tuyến?",
    "Tìm giúp tôi lời bài hát {song} của ca sĩ {singer}",
    "Cách pha cà phê sữa đá ngon đậm đà",
    "Cú pháp đăng ký gói cước internet mạng FPT theo tháng là gì?",
    "Cách làm sạch vết dầu mỡ cứng đầu bám trên quần áo",
    "Tìm rạp chiếu phim gần nhất ở {province}",
    "Điểm chuẩn trường đại học {university} năm ngoái là bao nhiêu?",
    "Cách khắc phục mạng wifi bị chập chờn liên tục tại nhà",
    "Cách phối đồ mùa đông cho nam trông thật phong cách và ấm áp",
    "Tải bộ hình nền máy tính 4K chủ đề thiên nhiên hoang dã",
    "Làm sao để tắt thông báo cập nhật phiền phức trên Windows 10?",
    "Tìm hiểu về lịch sử hình thành và phát triển của Vạn Lý Trường Thành",
    "Hướng dẫn tự học đệm hát guitar/piano cơ bản trong 30 ngày",
    "Làm thế nào để {tech_verb} trên hệ điều hành Linux?",
    "Làm sao để {tech_verb} mà không gặp lỗi bảo mật?",
    "Học {tech_topic} có khó hơn học {tech_topic} không?",
    "{subject} muốn tìm hiểu cách {cooking_verb} ngày cuối tuần",
    "{subject} đang gặp khó khăn khi {tech_verb} trong {tech_topic}"
]

# --- 4. TIẾN HÀNH SINH DỮ LIỆU ---
def generate_dataset(num_samples=3000):
    half_samples = num_samples // 2
    dataset = []

    # Sinh dữ liệu y tế (Nhãn 1)
    seen_medical = set()
    attempts = 0
    while len(dataset) < half_samples and attempts < 100000:
        attempts += 1
        temp = random.choice(medical_templates)
        sentence = temp.format(
            symptom=random.choice(symptoms),
            disease=random.choice(diseases),
            doctor=random.choice(doctors),
            specialty=random.choice(specialties),
            province=random.choice(provinces),
            hospital=random.choice(hospitals),
            procedure=random.choice(procedures),
            day=random.choice(days),
            time=random.choice(times)
        )
        if sentence not in seen_medical:
            seen_medical.add(sentence)
            dataset.append((sentence, 1))

    # Sinh dữ liệu ngoài lề (Nhãn 0)
    seen_non_medical = set()
    attempts = 0
    while len(dataset) < num_samples and attempts < 100000:
        attempts += 1
        temp = random.choice(non_medical_templates)
        sentence = temp.format(
            province=random.choice(provinces),
            tech_topic=random.choice(tech_topics),
            tech_verb=random.choice(tech_verbs),
            cooking_verb=random.choice(cooking_verbs),
            location=random.choice(locations),
            song=random.choice(songs),
            singer=random.choice(singers),
            game=random.choice(games),
            university=random.choice(universities),
            subject=random.choice(subjects)
        )
        if sentence not in seen_non_medical:
            seen_non_medical.add(sentence)
            dataset.append((sentence, 0))

    # Tráo ngẫu nhiên thứ tự các câu hỏi
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    target_file = "medical_dataset.csv"
    print("Generating 3000 dataset samples...")
    data = generate_dataset(3000)
    
    with open(target_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"]) # Header
        writer.writerows(data)
        
    print("Success! Wrote " + str(len(data)) + " dataset samples into " + target_file)
