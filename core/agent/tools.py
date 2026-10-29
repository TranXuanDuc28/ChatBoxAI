import logging
from core.data.web_data_manager import WebDataManager

logger = logging.getLogger("enhanced_medical_chatbot")
data_manager = WebDataManager()

def check_doctor_schedule(doctor_name: str, date_query: str = "Hôm nay") -> str:
    """
    Tra cứu lịch khám bệnh, địa chỉ phòng khám, số điện thoại liên hệ và giá khám của một bác sĩ cụ thể theo ngày.
    Args:
        doctor_name: Tên của bác sĩ cần tra cứu lịch khám (ví dụ: 'Nam', 'Trần Xuân Đức').
        date_query: Ngày muốn kiểm tra lịch (ví dụ: 'Hôm nay', 'Ngày mai').
    Returns:
        Thông tin chi tiết về bác sĩ và lịch trình khám của họ.
    """
    try:
        web_data = data_manager.get_web_data()
        doctors = web_data.get("doctors", [])
        
        query_name = doctor_name.lower()
        matched_doctors = []
        for doc in doctors:
            doc_full_name = f"{doc.get('firstName', '')} {doc.get('lastName', '')}".strip().lower()
            if query_name in doc_full_name or doc_full_name in query_name:
                matched_doctors.append(doc)
                
        if not matched_doctors:
            return f"Không tìm thấy bác sĩ nào có tên khớp với '{doctor_name}' trong hệ thống."
            
        response_lines = []
        for doc in matched_doctors:
            full_name = f"{doc.get('firstName', '')} {doc.get('lastName', '')}".strip()
            schedules = doc.get("schedules", [])
            schedule_str = "Chưa có lịch khám cụ thể được cập nhật."
            if schedules:
                target_scheds = [s for s in schedules if date_query.lower() in s.lower()]
                if target_scheds:
                    schedule_str = "\n".join(target_scheds)
                else:
                    schedule_str = f"Không có lịch khám vào ngày {date_query}. Các lịch khám hiện có:\n" + "\n".join(schedules)
                    
            response_lines.append(
                f"👨‍⚕️ Bác sĩ: {full_name}\n"
                f"- Chuyên khoa: {doc.get('specialty', 'N/A')}\n"
                f"- Phòng khám: {doc.get('clinic', 'N/A')}\n"
                f"- Lịch khám ({date_query}): {schedule_str}\n"
                f"- Giá khám: {doc.get('price', 'N/A')} VND\n"
                f"- Địa chỉ phòng khám: {doc.get('addressClinic', 'N/A')}\n"
                f"- SĐT liên hệ: {doc.get('phoneNumber', 'N/A')}"
            )
            
        return "\n\n".join(response_lines)
    except Exception as e:
        logger.error(f"Error in tool check_doctor_schedule: {e}")
        return f"Lỗi xảy ra khi tra cứu lịch khám của bác sĩ: {str(e)}"

def check_doctors_by_specialty(specialty_name: str) -> str:
    """
    Tra cứu danh sách tất cả các bác sĩ thuộc một chuyên khoa cụ thể.
    Args:
        specialty_name: Tên chuyên khoa cần tìm bác sĩ (ví dụ: 'Da liễu', 'Cơ xương khớp').
    Returns:
        Danh sách bác sĩ thuộc chuyên khoa đó kèm theo phòng khám và giá khám.
    """
    try:
        web_data = data_manager.get_web_data()
        doctors = web_data.get("doctors", [])
        
        query_spec = specialty_name.lower()
        matched_doctors = []
        for doc in doctors:
            doc_spec = doc.get("specialty", "").lower()
            if query_spec in doc_spec or doc_spec in query_spec:
                matched_doctors.append(doc)
                
        if not matched_doctors:
            return f"Không tìm thấy bác sĩ nào thuộc chuyên khoa '{specialty_name}'."
            
        response_lines = [f"Danh sách bác sĩ thuộc chuyên khoa {specialty_name}:"]
        for i, doc in enumerate(matched_doctors, 1):
            full_name = f"{doc.get('firstName', '')} {doc.get('lastName', '')}".strip()
            response_lines.append(
                f"{i}. Bác sĩ {full_name}\n"
                f"   - Phòng khám: {doc.get('clinic', 'N/A')}\n"
                f"   - Tỉnh thành: {doc.get('province', 'N/A')}\n"
                f"   - Giá khám: {doc.get('price', 'N/A')} VND"
            )
            
        return "\n".join(response_lines)
    except Exception as e:
        logger.error(f"Error in tool check_doctors_by_specialty: {e}")
        return f"Lỗi xảy ra khi tìm bác sĩ theo chuyên khoa: {str(e)}"

def check_clinic_info(clinic_name: str) -> str:
    """
    Tra cứu thông tin chi tiết (địa chỉ, giới thiệu và dịch vụ) của một bệnh viện hoặc phòng khám cụ thể.
    Args:
        clinic_name: Tên phòng khám hoặc bệnh viện cần tra cứu (ví dụ: 'Bệnh viện Da liễu', 'Bệnh viện Trung ương').
    Returns:
        Thông tin địa chỉ và giới thiệu chi tiết về phòng khám/bệnh viện.
    """
    try:
        web_data = data_manager.get_web_data()
        clinics = web_data.get("clinics", [])
        
        query_clinic = clinic_name.lower()
        matched_clinics = []
        for c in clinics:
            # Check clinicData or clinic itself if it's flat
            name = ""
            c_data = c.get("clinicData", {}) if isinstance(c, dict) else {}
            if c_data:
                name = c_data.get("name", "").lower()
            else:
                name = c.get("name", "").lower()
                
            if query_clinic in name or name in query_clinic:
                matched_clinics.append(c)
                
        if not matched_clinics:
            return f"Không tìm thấy phòng khám/bệnh viện nào khớp với tên '{clinic_name}'."
            
        response_lines = []
        for c in matched_clinics:
            c_data = c.get("clinicData", {})
            c_md = c.get("clinicMarkdown", {})
            
            name = c_data.get("name", c.get("name", "N/A"))
            address = c_data.get("address", c.get("address", "N/A"))
            desc = c_data.get("description", "")
            content = c_md.get("contentMarkdown", "")
            
            info = f"🏥 Phòng khám/Bệnh viện: {name}\n- Địa chỉ: {address}\n"
            if desc:
                info += f"- Giới thiệu: {desc[:300]}...\n"
            if content:
                info += f"- Dịch vụ & thông tin thêm: {content[:400]}...\n"
            response_lines.append(info)
            
        return "\n\n".join(response_lines)
    except Exception as e:
        logger.error(f"Error in tool check_clinic_info: {e}")
        return f"Lỗi xảy ra khi tra cứu thông tin phòng khám: {str(e)}"

def check_packages(keyword: str = "") -> str:
    """
    Tra cứu danh sách các gói khám sức khỏe tổng quát, tầm soát bệnh lý và giá niêm yết.
    Args:
        keyword: Từ khóa tìm gói khám (ví dụ: 'tổng quát', 'tim mạch', 'vip').
    Returns:
        Thông tin chi tiết các gói khám phù hợp.
    """
    try:
        web_data = data_manager.get_web_data()
        packages = web_data.get("packages", [])
        if not packages:
            return "Hiện tại hệ thống chưa cập nhật danh sách gói khám."

        query = keyword.strip().lower()
        matched = []
        for pkg in packages:
            name = (pkg.get("name") or "").lower()
            cat = (pkg.get("category") or "").lower()
            if not query or query in name or query in cat:
                matched.append(pkg)

        if not matched:
            matched = packages[:3]

        lines = ["Danh mục gói khám sức khỏe từ hệ thống:"]
        for p in matched[:5]:
            raw_price = p.get("price")
            try:
                price_str = f"{float(raw_price):,.0f} VNĐ" if raw_price else "Liên hệ"
            except:
                price_str = str(raw_price)
            lines.append(
                f"📦 **Gói khám: {p.get('name')}**\n"
                f"- Phân loại: {p.get('category', 'Tổng quát')}\n"
                f"- Giá trọn gói: {price_str}\n"
                f"- Độ tuổi: {p.get('ageRange', 'Mọi lứa tuổi')}\n"
                f"- Chi tiết: {p.get('description', '')[:200]}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        logger.error(f"Error in tool check_packages: {e}")
        return f"Lỗi xảy ra khi tra cứu gói khám: {str(e)}"

def check_medicine_info(medicine_name: str) -> str:
    """
    Tra cứu thông tin danh mục thuốc, công dụng và đơn giá niêm yết trong hệ thống.
    Args:
        medicine_name: Tên thuốc cần tra cứu (ví dụ: 'Paracetamol', 'Panadol').
    Returns:
        Thông tin về thuốc và đơn vị tính.
    """
    try:
        web_data = data_manager.get_web_data()
        medicines = web_data.get("medicines", [])
        if not medicines:
            return "Hệ thống chưa có dữ liệu dược phẩm."

        query = medicine_name.strip().lower()
        matched = []
        for m in medicines:
            name = (m.get("name") or "").lower()
            if query in name:
                matched.append(m)

        if not matched:
            return f"Không tìm thấy thuốc khớp với tên '{medicine_name}' trong danh mục."

        lines = []
        for m in matched[:3]:
            price = m.get("price", 0)
            try:
                price_str = f"{float(price):,.0f} VNĐ/{m.get('unit', 'đơn vị')}" if price else "Niêm yết tại quầy"
            except:
                price_str = str(price)
            lines.append(
                f"💊 **Thuốc: {m.get('name')}**\n"
                f"- Đơn vị tính: {m.get('unit', 'Hộp')}\n"
                f"- Giá niêm yết: {price_str}\n"
                f"- Công dụng: {m.get('description', 'Chưa có mô tả chi tiết')}\n"
                f"*Lưu ý: Bệnh nhân chỉ sử dụng thuốc khi có đơn kê của bác sĩ.*"
            )
        return "\n\n".join(lines)
    except Exception as e:
        logger.error(f"Error in tool check_medicine_info: {e}")
        return f"Lỗi xảy ra khi tra cứu thuốc: {str(e)}"

