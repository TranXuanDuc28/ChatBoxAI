import re
import unicodedata
from typing import Dict, List, Optional, Tuple

def remove_diacritics(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để so khớp không phân biệt dấu/chính tả."""
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text.lower().strip()

# Danh mục đầy đủ 63 Tỉnh/Thành phố Việt Nam kèm bí danh phổ biến
VIETNAM_PROVINCES: Dict[str, List[str]] = {
    # Miền Bắc
    "Hà Nội": ["ha noi", "hn", "hnoi", "thu do", "ha loi"],
    "Hải Phòng": ["hai phong", "hp", "dat cang"],
    "Quảng Ninh": ["quang ninh", "ha long", "cam pha", "uong bi", "mong cai"],
    "Bắc Ninh": ["bac ninh"],
    "Bắc Giang": ["bac giang"],
    "Hải Dương": ["hai duong"],
    "Hưng Yên": ["hung yen"],
    "Thái Bình": ["thai binh"],
    "Nam Định": ["nam dinh"],
    "Ninh Bình": ["ninh binh", "trang an"],
    "Hà Nam": ["ha nam", "phu ly"],
    "Vĩnh Phúc": ["vinh phuc", "vinh yen"],
    "Phú Thọ": ["phu tho", "viet tri", "den hung"],
    "Thái Nguyên": ["thai nguyen"],
    "Tuyên Quang": ["tuyen quang"],
    "Lạng Sơn": ["lang son"],
    "Cao Bằng": ["cao bang"],
    "Bắc Kạn": ["bac kan"],
    "Hà Giang": ["ha giang", "dong van"],
    "Lào Cai": ["lao cai", "sapa", "sa pa"],
    "Yên Bái": ["yen bai"],
    "Sơn La": ["son la", "moc chau"],
    "Hòa Bình": ["hoa binh"],
    "Điện Biên": ["dien bien", "dien bien phu"],
    "Lai Châu": ["lai chau"],

    # Miền Trung & Tây Nguyên
    "Thanh Hóa": ["thanh hoa", "sam son"],
    "Nghệ An": ["nghe an", "tp vinh", "cua lo"],
    "Hà Tĩnh": ["ha tinh"],
    "Quảng Bình": ["quang binh", "dong hoi", "phong nha"],
    "Quảng Trị": ["quang tri", "dong ha"],
    "Thừa Thiên Huế": ["thua thien hue", "hue", "tp hue", "co do hue"],
    "Đà Nẵng": ["da nang", "danang", "dn", "thanh pho dang song"],
    "Quảng Nam": ["quang nam", "hoi an", "tam ky"],
    "Quảng Ngãi": ["quang ngai"],
    "Bình Định": ["binh dinh", "quy nhon"],
    "Phú Yên": ["phu yen", "tuy hoa"],
    "Khánh Hòa": ["khanh hoa", "nha trang", "cam ranh"],
    "Ninh Thuận": ["ninh thuan", "phan rang"],
    "Bình Thuận": ["binh thuan", "phan thiet", "mui ne"],
    "Kon Tum": ["kon tum"],
    "Gia Lai": ["gia lai", "pleiku"],
    "Đắk Lắk": ["dak lak", "dac lac", "buon ma thuot", "bmt"],
    "Đắk Nông": ["dak nong", "gia nghia"],
    "Lâm Đồng": ["lam dong", "da lat", "dalat", "bao loc"],

    # Miền Nam
    "Hồ Chí Minh": ["ho chi minh", "tphcm", "tp hcm", "tp.hcm", "hcm", "sai gon", "saigon", "sai thanh"],
    "Bình Dương": ["binh duong", "thu dau mot", "di an", "thuan an", "ben cat", "tan uyen"],
    "Đồng Nai": ["dong nai", "bien hoa", "long khanh", "long thanh", "nhon trach"],
    "Bà Rịa - Vũng Tàu": ["ba ria vung tau", "vung tau", "ba ria", "con dao"],
    "Tây Ninh": ["tay ninh"],
    "Bình Phước": ["binh phuoc", "dong xoai"],
    "Long An": ["long an", "tan an"],
    "Tiền Giang": ["tien giang", "my tho"],
    "Bến Tre": ["ben tre"],
    "Trà Vinh": ["tra vinh"],
    "Vĩnh Long": ["vinh long"],
    "Đồng Tháp": ["dong thap", "cao lanh", "sa dec"],
    "An Giang": ["an giang", "long xuyen", "chau doc"],
    "Kiên Giang": ["kien giang", "rach gia", "ha tien", "phu quoc"],
    "Cần Thơ": ["can tho", "tay do", "ninh kieu"],
    "Hậu Giang": ["hau giang", "vi thanh"],
    "Sóc Trăng": ["soc trang"],
    "Bạc Liêu": ["bac lieu"],
    "Cà Mau": ["ca mau"]
}

# Danh mục địa danh, công trình, mốc giao thông nổi tiếng và khu vực tương ứng
POPULAR_LANDMARKS: Dict[str, Tuple[str, Optional[str]]] = {
    # (Tỉnh/Thành, Quận/Huyện)
    # TP.HCM
    "landmark 81": ("Hồ Chí Minh", "Bình Thạnh"),
    "hang xanh": ("Hồ Chí Minh", "Bình Thạnh"),
    "ben xe mien dong": ("Hồ Chí Minh", "Bình Thạnh"),
    "ben xe mien tay": ("Hồ Chí Minh", "Bình Tân"),
    "cho ben thanh": ("Hồ Chí Minh", "Quận 1"),
    "pho di bo nguyen hue": ("Hồ Chí Minh", "Quận 1"),
    "ho con rua": ("Hồ Chí Minh", "Quận 3"),
    "khu sala": ("Hồ Chí Minh", "Thủ Đức"),
    "khu cong nghe cao": ("Hồ Chí Minh", "Thủ Đức"),
    "suoi tien": ("Hồ Chí Minh", "Thủ Đức"),
    "cho lon": ("Hồ Chí Minh", "Quận 5"),
    "cau anh sao": ("Hồ Chí Minh", "Quận 7"),
    "phu my hung": ("Hồ Chí Minh", "Quận 7"),
    "san bay tan son nhat": ("Hồ Chí Minh", "Tân Bình"),

    # Hà Nội
    "ho guom": ("Hà Nội", "Hoàn Kiếm"),
    "ho hoan kiem": ("Hà Nội", "Hoàn Kiếm"),
    "pho co": ("Hà Nội", "Hoàn Kiếm"),
    "cho dong xuan": ("Hà Nội", "Hoàn Kiếm"),
    "ho tay": ("Hà Nội", "Tây Hồ"),
    "san bay noi bai": ("Hà Nội", "Sóc Sơn"),
    "ben xe my dinh": ("Hà Nội", "Nam Từ Liêm"),
    "ben xe giap bat": ("Hà Nội", "Hoàng Mai"),
    "ben xe nuoc ngam": ("Hà Nội", "Hoàng Mai"),
    "keangnam": ("Hà Nội", "Nam Từ Liêm"),
    "trung hoa nhan chinh": ("Hà Nội", "Cầu Giấy"),

    # Đà Nẵng
    "cau rong": ("Đà Nẵng", "Hải Châu"),
    "cau song han": ("Đà Nẵng", "Hải Châu"),
    "ba na hills": ("Đà Nẵng", "Hòa Vang"),
    "ban dao son tra": ("Đà Nẵng", "Sơn Trà"),
    "ngu hanh son": ("Đà Nẵng", "Ngũ Hành Sơn"),

    # Khác
    "kcn vsip": ("Bình Dương", "Thuận An"),
    "song than": ("Bình Dương", "Dĩ An"),
    "kcn amata": ("Đồng Nai", "Biên Hòa"),
    "ho xuan huong": ("Lâm Đồng", "Đà Lạt")
}

# Danh mục các quận/huyện phổ biến
POPULAR_DISTRICTS: Dict[str, Dict[str, List[str]]] = {
    "Hồ Chí Minh": {
        "Quận 1": ["quan 1", "q1", "q.1", "district 1"],
        "Quận 2": ["quan 2", "q2", "q.2", "district 2"],
        "Quận 3": ["quan 3", "q3", "q.3", "district 3"],
        "Quận 4": ["quan 4", "q4", "q.4", "district 4"],
        "Quận 5": ["quan 5", "q5", "q.5", "district 5"],
        "Quận 6": ["quan 6", "q6", "q.6", "district 6"],
        "Quận 7": ["quan 7", "q7", "q.7", "district 7"],
        "Quận 8": ["quan 8", "q8", "q.8", "district 8"],
        "Quận 9": ["quan 9", "q9", "q.9", "district 9"],
        "Quận 10": ["quan 10", "q10", "q.10", "district 10"],
        "Quận 11": ["quan 11", "q11", "q.11", "district 11"],
        "Quận 12": ["quan 12", "q12", "q.12", "district 12"],
        "Thủ Đức": ["thu duc", "tp thu duc", "thanh pho thu duc"],
        "Bình Thạnh": ["binh thanh"],
        "Gò Vấp": ["go vap"],
        "Phú Nhuận": ["phu nhuan"],
        "Tân Bình": ["tan binh"],
        "Tân Phú": ["tan phu"],
        "Bình Tân": ["binh tan"],
        "Củ Chi": ["cu chi"],
        "Hóc Môn": ["hoc mon"],
        "Bình Chánh": ["binh chanh"],
        "Nhà Bè": ["nha be"],
        "Cần Giờ": ["can gio"]
    },
    "Hà Nội": {
        "Ba Đình": ["ba dinh"],
        "Hoàn Kiếm": ["hoan kiem"],
        "Tây Hồ": ["tay ho"],
        "Long Biên": ["long bien"],
        "Cầu Giấy": ["cau giay"],
        "Đống Đa": ["dong da"],
        "Hai Bà Trưng": ["hai ba trung"],
        "Hoàng Mai": ["hoang mai"],
        "Thanh Xuân": ["thanh xuan"],
        "Nam Từ Liêm": ["nam tu liem"],
        "Bắc Từ Liêm": ["bac tu liem"],
        "Hà Đông": ["ha dong"],
        "Sơn Tây": ["son tay"],
        "Ba Vì": ["ba vi"],
        "Đông Anh": ["dong anh"],
        "Gia Lâm": ["gia lam"],
        "Hoài Đức": ["hoai duc"],
        "Mê Linh": ["me linh"],
        "Sóc Sơn": ["soc son"],
        "Thạch Thất": ["thach that"],
        "Thanh Trì": ["thanh tri"],
        "Thường Tín": ["thuong tin"],
        "Chương Mỹ": ["chuong my"],
        "Đan Phượng": ["dan phuong"],
        "Quốc Oai": ["quoc oai"]
    },
    "Đà Nẵng": {
        "Hải Châu": ["hai chau"],
        "Thanh Khê": ["thanh khe"],
        "Sơn Trà": ["son tra"],
        "Ngũ Hành Sơn": ["ngu hanh son"],
        "Liên Chiểu": ["lien chieu"],
        "Cẩm Lệ": ["cam le"],
        "Hòa Vang": ["hoa vang"]
    }
}

class GeographicResolver:
    """
    Bộ giải mã và định vị địa lý thông minh đa tầng:
    1. Tự động đồng bộ các tỉnh/địa chỉ thực tế từ Database.
    2. Nhận diện cả 63 tỉnh thành Việt Nam với mọi biến thể viết tắt/không dấu/teencode.
    3. Tự suy luận quận/huyện dựa trên địa danh, công trình nổi tiếng (Zero-shot Landmarks).
    4. Cung cấp fallback linh hoạt cho LLM khi gặp địa danh nằm ngoài danh mục.
    """
    def __init__(self, db_provinces: Optional[List[str]] = None):
        self.provinces_catalog = dict(VIETNAM_PROVINCES)
        if db_provinces:
            self.update_db_provinces(db_provinces)

    def update_db_provinces(self, db_provinces: List[str]):
        """Cập nhật các tỉnh thành thực tế từ Database."""
        for p in db_provinces:
            if not p or not isinstance(p, str):
                continue
            p_clean = p.strip()
            norm = remove_diacritics(p_clean)
            if p_clean not in self.provinces_catalog:
                self.provinces_catalog[p_clean] = [norm]
            else:
                if norm not in self.provinces_catalog[p_clean]:
                    self.provinces_catalog[p_clean].append(norm)

    def resolve_location(self, message: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Phân tích chuỗi đầu vào và trả về bộ ba:
        (full_display_location, target_province, target_district)
        """
        if not message or not str(message).strip():
            return None, None, None

        msg_raw = str(message).strip()
        msg_norm = remove_diacritics(msg_raw)

        detected_province = None
        detected_district = None

        # 1. Kiểm tra mốc địa danh / Landmark nổi tiếng (Zero-shot landmark reasoning)
        for landmark, (prov, dist) in POPULAR_LANDMARKS.items():
            pattern = r'(?:\b|^)' + re.escape(landmark) + r'(?:\b|$)'
            if re.search(pattern, msg_norm):
                detected_province = prov
                detected_district = dist
                break

        # 2. Nếu chưa thấy landmark, kiểm tra quận/huyện theo từng tỉnh
        if not detected_district:
            for prov_name, dist_dict in POPULAR_DISTRICTS.items():
                for dist_name, aliases in dist_dict.items():
                    for alias in aliases:
                        pattern = r'(?:\b|^)' + re.escape(alias) + r'(?:\b|$)'
                        if re.search(pattern, msg_norm):
                            detected_district = dist_name
                            detected_province = prov_name
                            break
                    if detected_district:
                        break
                if detected_district:
                    break

        # 3. Kiểm tra tỉnh/thành phố trong toàn bộ 63 tỉnh thành Việt Nam
        if not detected_province:
            for canonical_name, aliases in self.provinces_catalog.items():
                # Kiểm tra tên chính thức
                canon_norm = remove_diacritics(canonical_name)
                pattern_canon = r'(?:\b|^)' + re.escape(canon_norm) + r'(?:\b|$)'
                if re.search(pattern_canon, msg_norm):
                    detected_province = canonical_name
                    break
                # Kiểm tra bí danh
                matched = False
                for alias in aliases:
                    pattern_alias = r'(?:\b|^)' + re.escape(alias) + r'(?:\b|$)'
                    if re.search(pattern_alias, msg_norm):
                        detected_province = canonical_name
                        matched = True
                        break
                if matched:
                    break

        # 4. Tạo nhãn hiển thị đầy đủ
        if detected_district and detected_province:
            prov_display = "TP. Hồ Chí Minh" if detected_province == "Hồ Chí Minh" else detected_province
            full_display = f"{detected_district}, {prov_display}"
        elif detected_province:
            full_display = "TP. Hồ Chí Minh" if detected_province == "Hồ Chí Minh" else detected_province
        elif detected_district:
            full_display = detected_district
        else:
            full_display = None

        return full_display, detected_province, detected_district

# Khởi tạo singleton mặc định
default_geo_resolver = GeographicResolver()
