import os
import requests
import time
import logging
from typing import Dict
from dotenv import load_dotenv
logger = logging.getLogger("enhanced_medical_chatbot")
load_dotenv()
BACKEND_API_BASE = os.getenv("BACKEND_API_BASE")
BACKEND_LANG = os.getenv("BACKEND_LANG")

class WebDataManager:
    def __init__(self):
        self.cache_duration = 600
        self.last_update = 0
        self.cached_data = None
    
    def get_web_data(self) -> Dict:
        current_time = time.time()
        if (self.cached_data is None or 
            current_time - self.last_update > self.cache_duration):
            logger.info("Fetching fresh web data...")
            self.cached_data = self._fetch_web_data()
            self.last_update = current_time
        return self.cached_data
    
    def _fetch_web_data(self) -> dict:
        web_data = {
            "clinics": [],
            "specialties": [],
            "doctors": [],
            "handbooks": []
        }

        apis_to_call = [
            ("clinics", f"{BACKEND_API_BASE}/get-clinic?lang={BACKEND_LANG}"),
            ("specialties", f"{BACKEND_API_BASE}/get-specialty?lang={BACKEND_LANG}"),
            ("handbooks", f"{BACKEND_API_BASE}/get-handbook?lang={BACKEND_LANG}")
        ]

        for data_type, url in apis_to_call:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("errCode") == 0:
                        items = data.get("data", [])[:10]  # Lấy tối đa 8 item
                        # Loại bỏ image nếu có
                        for item in items:
                            if "image" in item:
                                item.pop("image")
                        web_data[data_type] = items
                        logger.info(f"Fetched {len(items)} {data_type}")
                    else:
                        logger.warning(f"API error for {url}: {data.get('errMessage', 'Unknown')}")
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching {data_type} from {url}")
            except Exception as e:
                logger.warning(f"Error fetching {data_type}: {str(e)}")

        # Lấy doctors
        try:
            doctor_response = requests.get(f"{BACKEND_API_BASE}/get_all_doctor", timeout=5)
            if doctor_response.status_code == 200:
                doctor_data = doctor_response.json()
                if doctor_data.get("errCode") == 0:
                    doctors = doctor_data.get("data", [])[:10]  # Lấy tối đa 10 doctor
                    detailed_doctors = []

                    for doctor in doctors:
                        try:
                            detail_response = requests.get(
                                f"{BACKEND_API_BASE}/get-extra-doctor-by-id?doctorId={doctor['id']}&lang={BACKEND_LANG}",
                                timeout=3
                            )
                            if detail_response.status_code == 200:
                                detail_data = detail_response.json()
                                if detail_data.get("errCode") == 0:
                                    doctor_detail = detail_data.get("data", {})  # Đây là 1 object duy nhất
                                    # Loại bỏ image nếu có
                                    doctor_detail.pop("image", None)

                                    detailed_doctor = {
                                        "id": doctor.get("id"),
                                        "firstName": doctor.get("firstName", ""),
                                        "lastName": doctor.get("lastName", ""),
                                        "specialty": doctor_detail.get("specialty", {}).get("name", ""),
                                        "clinic": doctor_detail.get("nameClinic", doctor_detail.get("addressClinic", "N/A")),
                                        "note": doctor_detail.get("note", "")[:100],
                                        "price": doctor_detail.get("priceTypeData", {}).get("valueVi", ""),
                                        "payment": doctor_detail.get("paymentTypeData", {}).get("valueVi", ""),
                                        "province": doctor_detail.get("provinceTypeData", {}).get("valueVi", "")
                                    }
                                    detailed_doctors.append(detailed_doctor)
                                    print("detailed_doctor", detailed_doctor)
                        except Exception as e:
                            logger.warning(f"Error fetching doctor {doctor.get('id')}: {str(e)}")
                            continue

                    web_data["doctors"] = detailed_doctors
                    logger.info(f"Fetched {len(detailed_doctors)} doctors")
        except Exception as e:
            logger.warning(f"Error fetching doctors: {str(e)}")


        return web_data

    


   