import os
import requests
import time
import logging
from typing import Dict, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

logger = logging.getLogger("enhanced_medical_chatbot")
load_dotenv()
BACKEND_API_BASE = os.getenv("BACKEND_API_BASE", "http://localhost:5000/api")
BACKEND_LANG = os.getenv("BACKEND_LANG", "vi")

class WebDataManager:
    def __init__(self):
        self.cache_duration = 600
        self.last_update = 0
        self.cached_data = None
    
    def get_web_data(self) -> Dict:
        current_time = time.time()
        if (self.cached_data is None or 
            current_time - self.last_update > self.cache_duration):
            logger.info("Fetching fresh web data from backend...")
            self.cached_data = self._fetch_web_data()
            self.last_update = current_time
        return self.cached_data
    
    def _fetch_web_data(self) -> dict:
        web_data = {
            "clinics": [],
            "specialties": [],
            "doctors": [],
            "handbooks": [],
            "packages": [],
            "facilities": [],
            "medicines": [],
            "provinces": []
        }

        # 1. Các API danh mục cơ bản
        apis_to_call = [
            ("clinics", f"{BACKEND_API_BASE}/get-clinic?lang={BACKEND_LANG}"),
            ("specialties", f"{BACKEND_API_BASE}/get-specialty?lang={BACKEND_LANG}"),
            ("handbooks", f"{BACKEND_API_BASE}/get-handbook?lang={BACKEND_LANG}"),
            ("packages", f"{BACKEND_API_BASE}/packages/all"),
            ("facilities", f"{BACKEND_API_BASE}/facilities/all"),
            ("medicines", f"{BACKEND_API_BASE}/get-medicines"),
            ("provinces", f"{BACKEND_API_BASE}/allcode?type=PROVINCE")
        ]

        for data_type, url in apis_to_call:
            try:
                response = requests.get(url, timeout=6)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("errCode") == 0:
                        items = data.get("data", [])
                        # Loại bỏ các trường hình ảnh nặng để giảm dung lượng
                        for item in items:
                            if isinstance(item, dict):
                                item.pop("image", None)
                                item.pop("logo", None)
                                item.pop("images", None)
                        web_data[data_type] = items
                        logger.info(f"✅ Fetched {len(items)} {data_type}")
                    else:
                        logger.warning(f"API error for {url}: {data.get('errMessage', 'Unknown')}")
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching {data_type} from {url}")
            except Exception as e:
                logger.warning(f"Error fetching {data_type}: {str(e)}")

        # 2. Lấy dữ liệu Bác sĩ chi tiết và Lịch khám 7 ngày tới
        try:
            doctor_response = requests.get(f"{BACKEND_API_BASE}/get_all_doctor", timeout=10)
            if doctor_response.status_code == 200:
                doctor_data = doctor_response.json()
                if doctor_data.get("errCode") == 0:
                    doctors = doctor_data.get("data", [])
                    detailed_doctors = []

                    # Chuẩn bị dữ liệu lịch khám 7 ngày liên tiếp
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    day_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
                    days_to_check = []
                    for i in range(7):
                        target_day = today + timedelta(days=i)
                        target_ts = int(target_day.timestamp() * 1000)
                        weekday_name = day_names[target_day.weekday()]
                        if i == 0:
                            label = f"Hôm nay ({weekday_name}, {target_day.strftime('%d/%m')})"
                        elif i == 1:
                            label = f"Ngày mai ({weekday_name}, {target_day.strftime('%d/%m')})"
                        else:
                            label = f"{weekday_name} ({target_day.strftime('%d/%m')})"
                        days_to_check.append((label, target_ts, target_day))

                    def fetch_single_doctor_data(doctor):
                        try:
                            doc_id = doctor.get('id')
                            if not doc_id:
                                return None

                            # 1. Chi tiết thông tin bác sĩ
                            detail_url = f"{BACKEND_API_BASE}/get-extra-doctor-by-id?doctorId={doc_id}&lang={BACKEND_LANG}"
                            det_resp = requests.get(detail_url, timeout=5)
                            doctor_detail = {}
                            if det_resp.status_code == 200:
                                det_data = det_resp.json()
                                if det_data.get("errCode") == 0:
                                    doctor_detail = det_data.get("data", {})
                                    doctor_detail.pop("image", None)

                            # 2. Lịch khám trong 7 ngày tới (ghi nhận rõ ràng từng ngày: có lịch hay chưa có lịch)
                            schedules = []
                            for label, ts, target_day in days_to_check:
                                try:
                                    sch_url = f"{BACKEND_API_BASE}/get_schedule_doctor_by_date?doctorId={doc_id}&date={ts}"
                                    sch_resp = requests.get(sch_url, timeout=3)
                                    if sch_resp.status_code == 200:
                                        sch_json = sch_resp.json()
                                        if sch_json.get("errCode") == 0 and sch_json.get("data"):
                                            slots = [s.get("timeTypeData", {}).get("valueVi", "") for s in sch_json.get("data", [])]
                                            slots = [sl for sl in slots if sl]
                                            if slots:
                                                schedules.append(f"- {label}: {', '.join(slots)} (Có lịch)")
                                            else:
                                                schedules.append(f"- {label}: Chưa có lịch khám trực tuyến")
                                        else:
                                            schedules.append(f"- {label}: Chưa có lịch khám trực tuyến")
                                    else:
                                        schedules.append(f"- {label}: Chưa có lịch khám trực tuyến")
                                except Exception:
                                    schedules.append(f"- {label}: Chưa có lịch khám trực tuyến")

                            first_name = doctor.get("firstName", "")
                            last_name = doctor.get("lastName", "")
                            full_name = f"{first_name} {last_name}".strip()

                            return {
                                "id": doc_id,
                                "name": full_name,
                                "firstName": first_name,
                                "lastName": last_name,
                                "phoneNumber": doctor.get("phoneNumber", ""),
                                "specialty": doctor_detail.get("specialty", {}).get("name", ""),
                                "clinic": doctor_detail.get("nameClinic", doctor_detail.get("addressClinic", "N/A")),
                                "addressClinic": doctor_detail.get("addressClinic", ""),
                                "note": (doctor_detail.get("note") or "")[:200],
                                "price": doctor_detail.get("priceTypeData", {}).get("valueVi", ""),
                                "payment": doctor_detail.get("paymentTypeData", {}).get("valueVi", ""),
                                "province": doctor_detail.get("provinceTypeData", {}).get("valueVi", ""),
                                "schedules": schedules
                            }
                        except Exception as e:
                            logger.error(f"Error fetching doctor {doctor.get('id')}: {e}")
                            return None

                    # Thực thi song song cho các bác sĩ
                    logger.info(f"Starting parallel fetch for {len(doctors)} doctors with 7-day schedules...")
                    with ThreadPoolExecutor(max_workers=10) as executor:
                        future_to_doctor = {executor.submit(fetch_single_doctor_data, doc): doc for doc in doctors}
                        for future in as_completed(future_to_doctor):
                            result = future.result()
                            if result:
                                detailed_doctors.append(result)

                    web_data["doctors"] = detailed_doctors

                    # 3. Lấy thêm chi tiết phòng khám (clinics)
                    detailed_clinics = []
                    def fetch_clinic_detail(clinic):
                        try:
                            clinic_id = clinic.get('id')
                            if not clinic_id:
                                return clinic
                            detail_url = f"{BACKEND_API_BASE}/get-detail-clinic-by-id?id={clinic_id}&lang={BACKEND_LANG}"
                            resp = requests.get(detail_url, timeout=5)
                            if resp.status_code == 200:
                                c_data = resp.json()
                                if c_data.get("errCode") == 0:
                                    res_clinic = c_data.get("data", clinic)
                                    res_clinic.pop("image", None)
                                    return res_clinic
                            return clinic
                        except Exception:
                            return clinic

                    logger.info(f"Starting parallel fetch for {len(web_data['clinics'])} clinics...")
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_clinic = {executor.submit(fetch_clinic_detail, c): c for c in web_data["clinics"]}
                        for future in as_completed(future_to_clinic):
                            res = future.result()
                            if res:
                                detailed_clinics.append(res)
                    
                    web_data["clinics"] = detailed_clinics
                    logger.info(f"✅ Successfully loaded {len(detailed_doctors)} doctors, {len(detailed_clinics)} clinics, "
                                f"{len(web_data['packages'])} packages, {len(web_data['facilities'])} facilities, "
                                f"{len(web_data['medicines'])} medicines")
        except Exception as e:
            logger.warning(f"Error fetching doctors/clinics details: {str(e)}")

        return web_data