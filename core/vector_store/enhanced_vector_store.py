import os
import faiss
import logging
import string
from typing import Dict, Optional, List, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

logger = logging.getLogger("enhanced_medical_chatbot")

def tokenize_vietnamese(text: str) -> List[str]:
    """Tokenize Vietnamese text simply by lowercasing and splitting words."""
    if not text:
        return []
    # Remove simple punctuation
    text_clean = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text_clean.split()

class EnhancedVectorStore:
    def __init__(self, embeddings, dimension: int = 384):
        self.embeddings = embeddings
        self.dimension = dimension
        self.vectorstore: Optional[Any] = None
        self.specialty_mapping = {}  # specialty_id -> specialty_info
        self.doctor_specialty_mapping = {}  # doctor_id -> specialty_info
        
        # BM25 Corpus for Hybrid Search
        self.raw_documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        
        # Qdrant configuration
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.use_qdrant = bool(self.qdrant_url)
        self.qdrant_client = None
        
        if self.use_qdrant:
            try:
                self.qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key
                )
                logger.info("✅ Qdrant client initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Qdrant client, falling back to FAISS: {e}")
                self.use_qdrant = False
    
    def build_from_web_data(self, web_data: Dict) -> None:
        """Xây dựng vector store từ dữ liệu web với liên kết thông minh."""
        if not web_data or not isinstance(web_data, dict):
            logger.error("Invalid web_data provided")
            self.vectorstore = None
            return

        # Bước 1: Xây dựng mapping cho specialties
        self._build_specialty_mapping(web_data.get("specialties", []))
        
        documents: List[Document] = []

        # Bước 2: Xử lý specialties với thông tin phong phú
        documents.extend(self._process_specialties(web_data.get("specialties", [])))
        
        # Bước 3: Xử lý doctors với liên kết chuyên khoa
        documents.extend(self._process_doctors_with_specialty_link(web_data.get("doctors", [])))
        
        # Bước 4: Xử lý clinics
        documents.extend(self._process_clinics(web_data.get("clinics", [])))
        
        # Bước 5: Xử lý handbooks
        documents.extend(self._process_handbooks(web_data.get("handbooks", [])))

        # Bước 6: Xử lý packages (Gói khám bệnh dịch vụ)
        documents.extend(self._process_packages(web_data.get("packages", [])))

        # Bước 7: Xử lý facilities (Cơ sở y tế / Bệnh viện đối tác)
        documents.extend(self._process_facilities(web_data.get("facilities", [])))

        # Bước 8: Xử lý medicines (Thuốc & Dược phẩm)
        documents.extend(self._process_medicines(web_data.get("medicines", [])))

        # Lưu documents thô và xây dựng BM25
        if documents:
            self.raw_documents = documents
            self._build_bm25_retriever(documents)
            
            try:
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                
                if self.use_qdrant and self.qdrant_client:
                    try:
                        self.vectorstore = Qdrant.from_documents(
                            documents=documents,
                            embedding=self.embeddings,
                            url=self.qdrant_url,
                            api_key=self.qdrant_api_key,
                            collection_name="medical_appointment_system",
                            force_recreate=True
                        )
                        logger.info("✅ Built vector store with Qdrant successfully")
                    except Exception as q_err:
                        logger.error(f"❌ Qdrant build failed, falling back to FAISS: {q_err}")
                        self._build_faiss(texts, metadatas)
                else:
                    self._build_faiss(texts, metadatas)
                    
                # Log số lượng theo loại document
                type_count = {}
                for d in documents:
                    t = d.metadata["type"]
                    type_count[t] = type_count.get(t, 0) + 1
                logger.info(f"Vector store breakdown: {type_count}")

            except Exception as e:
                logger.error(f"❌ Error building vector store: {str(e)}")
                self.vectorstore = None
        else:
            logger.warning("⚠️ No documents to build vector store")
            self.vectorstore = None

    def _build_faiss(self, texts: List[str], metadatas: List[Dict]) -> None:
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        logger.info(f"✅ Built vector store with FAISS successfully (Total docs: {len(texts)})")

    def _build_bm25_retriever(self, documents: List[Document]) -> None:
        try:
            tokenized_corpus = [tokenize_vietnamese(doc.page_content) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"✅ Initialized BM25 retriever with {len(documents)} documents")
        except Exception as e:
            logger.error(f"❌ Failed to build BM25 retriever: {e}")
            self.bm25 = None

    def _build_specialty_mapping(self, specialties: List[Dict]) -> None:
        """Xây dựng mapping cho các chuyên khoa."""
        for specialty in specialties:
            specialty_data = specialty.get("specialtyData", {})
            specialty_id = specialty_data.get("id")
            specialty_name = specialty_data.get("name", "")
            
            if specialty_id and specialty_name:
                # Lưu thông tin chuyên khoa với nhiều variant tên
                self.specialty_mapping[specialty_id] = {
                    "name": specialty_name,
                    "name_lower": specialty_name.lower(),
                    "keywords": self._extract_specialty_keywords(specialty_name),
                    "data": specialty_data
                }
                
                # Thêm mapping theo tên (để search theo tên)
                self.specialty_mapping[specialty_name.lower()] = self.specialty_mapping[specialty_id]

    def _extract_specialty_keywords(self, specialty_name: str) -> List[str]:
        """Trích xuất từ khóa từ tên chuyên khoa."""
        keywords = []
        name_lower = specialty_name.lower()
        
        # Mapping các chuyên khoa thường gặp
        specialty_keywords = {
            "tim": ["tim mạch", "cardiac", "cardiology", "heart"],
            "gan": ["gan mật", "hepatology", "liver"],
            "thần kinh": ["neurological", "neurology", "brain"],
            "nhi": ["pediatric", "children", "trẻ em"],
            "sản": ["obstetrics", "pregnancy", "thai nghén"],
            "phụ": ["gynecology", "women", "phụ nữ"],
            "mắt": ["ophthalmology", "eye", "vision"],
            "tai mũi họng": ["ent", "otolaryngology"],
            "da liễu": ["dermatology", "skin", "da"],
            "xương khớp": ["orthopedic", "bone", "joint", "xương", "khớp"],
        }
        
        # Tìm keywords phù hợp
        for key, values in specialty_keywords.items():
            if key in name_lower:
                keywords.extend(values)
        
        # Thêm chính tên chuyên khoa
        keywords.append(name_lower)
        return list(set(keywords))

    def _process_specialties(self, specialties: List[Dict]) -> List[Document]:
        """Xử lý specialties với thông tin phong phú."""
        documents = []
        
        for specialty in specialties:
            # specialtyData
            specialty_data = specialty.get("specialtyData")
            if isinstance(specialty_data, dict):
                name = specialty_data.get("name", "").strip()
                if name:
                    # Tạo text phong phú cho specialty
                    keywords = self._extract_specialty_keywords(name)
                    text = f"Chuyên khoa {name}. Liên quan đến: {', '.join(keywords[:5])}"
                    
                    entry_copy = {k: v for k, v in specialty_data.items() if k != "image"}
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "type": "specialty", 
                            "specialty_name": name,
                            "keywords": keywords,
                            "data": entry_copy
                        }
                    ))
                    logger.info(f"Added specialtyData: {name}")

            # specialtyMarkdown với liên kết
            specialty_md = specialty.get("specialtyMarkdown")
            if isinstance(specialty_md, dict):
                content = specialty_md.get("contentMarkdown", specialty_md.get("contentHTML", "")).strip()
                if content:
                    specialty_name = specialty_data.get("name", "") if specialty_data else ""
                    enhanced_content = f"Chuyên khoa {specialty_name}: {content}"
                    
                    md_copy = {k: v for k, v in specialty_md.items() if k != "image"}
                    documents.append(Document(
                        page_content=enhanced_content, 
                        metadata={
                            "type": "specialty_detail", 
                            "specialty_name": specialty_name,
                            "data": md_copy
                        }
                    ))
                    logger.info(f"Added specialtyMarkdown (length={len(content)} chars)")
                    
        return documents

    def _process_doctors_with_specialty_link(self, doctors: List[Dict]) -> List[Document]:
        """Xử lý doctors với liên kết chuyên khoa chi tiết."""
        documents = []
        
        for doctor in doctors:
            first_name = doctor.get('firstName', '').strip()
            last_name = doctor.get('lastName', '').strip()
            name = f"{first_name} {last_name}".strip()
            specialty_name = doctor.get("specialty", "N/A")
            clinic_name = doctor.get("clinic", "N/A")
            address_clinic = doctor.get("addressClinic", "")
            note = doctor.get("note", "")
            price = doctor.get("price", "")
            payment = doctor.get("payment", "")
            province = doctor.get("province", "")
            
            if name:
                # Lấy lịch khám
                schedule_info = ""
                schedules = doctor.get("schedules", [])
                if schedules:
                    schedule_info = "\n" + "\n".join(schedules)

                doc_id = doctor.get('id')
                booking_url = f"/detail-doctor/{doc_id}" if doc_id else ""

                # Tạo text phong phú và lặp lại tên bác sĩ để tăng trọng số tìm kiếm
                doctor_text = f"""Bác sĩ {name} (ID: {doc_id}) chuyên khoa {specialty_name} tại {clinic_name} ({province}).
Địa chỉ phòng khám / bệnh viện: {address_clinic if address_clinic else province}.
Giá khám: {price}. Thanh toán: {payment}.
Số điện thoại liên hệ: {doctor.get('phoneNumber', 'N/A')}.
Link đặt lịch trực tuyến trên website: {booking_url}
Lịch khám chi tiết 7 ngày tới của bác sĩ:
{schedule_info if schedule_info else 'Hiện tại chưa cập nhật lịch khám trực tuyến trên hệ thống. Vui lòng liên hệ phòng khám để biết chi tiết lịch khám cụ thể.'}
Ghi chú: {note[:200]}
Từ khóa: {name}, bác sĩ {name}, số điện thoại {name}, liên hệ {name}, lịch khám {name}, lịch khám ngày mai {name}, lịch khám hôm nay {name}, khung giờ {name}, {specialty_name}, {address_clinic}"""

                # Tìm keywords liên quan từ chuyên khoa
                specialty_keywords = []
                if specialty_name and specialty_name != "N/A":
                    specialty_keywords = self._extract_specialty_keywords(specialty_name)

                doctor_copy = {k: v for k, v in doctor.items() if k != "image"}
                documents.append(Document(
                    page_content=doctor_text.strip(), 
                    metadata={
                        "type": "doctor", 
                        "doctor_name": name,
                        "specialty_name": specialty_name,
                        "specialty_keywords": specialty_keywords,
                        "clinic_name": clinic_name,
                        "province": province,
                        "addressClinic": address_clinic,
                        "address": address_clinic,
                        "data": doctor_copy
                    }
                ))
                logger.info(f"Added doctor doc: {name} - {specialty_name}")
                
        return documents

    def _process_clinics(self, clinics: List[Dict]) -> List[Document]:
        """Xử lý clinics."""
        documents = []
        
        for clinic in clinics:
            # clinicData
            clinic_data = clinic.get("clinicData")
            if isinstance(clinic_data, dict):
                name = clinic_data.get("name", "").strip()
                description = clinic_data.get("description", "").strip()
                address = clinic_data.get("address", "").strip()
                if name or description:
                    text = f"Bệnh viện {name}\nĐịa chỉ: {address}\n{description}".strip()
                    data_copy = {k: v for k, v in clinic_data.items() if k != "image"}
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "type": "clinic", 
                            "clinic_name": name,
                            "address": address,
                            "province": address,
                            "data": data_copy
                        }
                    ))
                    logger.info(f"Added clinicData doc: {name}")

            # clinicMarkdown
            clinic_md = clinic.get("clinicMarkdown")
            if isinstance(clinic_md, dict):
                content = clinic_md.get("contentMarkdown", clinic_md.get("contentHTML", "")).strip()
                if content:
                    clinic_name = clinic.get('name', clinic_data.get('name', 'N/A') if clinic_data else 'N/A')
                    address = clinic_data.get('address', '') if clinic_data else ''
                    text = f"Bệnh viện {clinic_name}\nĐịa chỉ: {address}\n{content}".strip()
                    md_copy = {k: v for k, v in clinic_md.items() if k != "image"}
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "type": "clinic_detail", 
                            "clinic_name": clinic_name,
                            "address": address,
                            "province": address,
                            "data": md_copy
                        }
                    ))
                    logger.info(f"Added clinicMarkdown doc: {clinic_name} (length={len(content)} chars)")
                    
        return documents

    def _process_handbooks(self, handbooks: List[Dict]) -> List[Document]:
        """Xử lý handbooks."""
        documents = []
        
        for handbook in handbooks:
            # handbookData
            for entry in handbook.get("handbookData", []):
                title = entry.get("title", "").strip()
                if title:
                    text = f"Cẩm nang y tế: {title}"
                    entry_copy = {k: v for k, v in entry.items() if k != "image"}
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "type": "handbook", 
                            "title": title,
                            "data": entry_copy
                        }
                    ))
                    logger.info(f"Added handbookData: {title}")
                    
            # handbookMarkdown
            for md in handbook.get("handbookMarkdown", []):
                content = md.get("contentMarkdown", "").strip()
                if content:
                    md_copy = {k: v for k, v in md.items() if k != "image"}
                    documents.append(Document(
                        page_content=content, 
                        metadata={
                            "type": "handbook_detail", 
                            "data": md_copy
                        }
                    ))
                    logger.info(f"Added handbookMarkdown (length={len(content)} chars)")
                    
        return documents

    def _process_packages(self, packages: List[Dict]) -> List[Document]:
        """Xử lý các gói khám bệnh dịch vụ."""
        documents = []
        for pkg in packages:
            name = (pkg.get("name") or "").strip()
            if not name:
                continue
            desc = pkg.get("description", "") or ""
            category = pkg.get("category", "Tổng quát") or "Tổng quát"
            raw_price = pkg.get("price")
            try:
                price_str = f"{float(raw_price):,.0f} VNĐ" if raw_price is not None else "Liên hệ"
            except (ValueError, TypeError):
                price_str = str(raw_price) if raw_price else "Liên hệ"
                
            gender = pkg.get("gender", "both")
            gender_vn = "Cả nam và nữ" if gender == "both" else ("Nam giới" if gender == "male" else "Nữ giới")
            age_range = pkg.get("ageRange", "Mọi lứa tuổi") or "Mọi lứa tuổi"
            duration = pkg.get("duration", 60)
            
            facility_data = pkg.get("facilityData", {}) or {}
            facility_name = facility_data.get("name", "Cơ sở y tế đối tác") if isinstance(facility_data, dict) else "Cơ sở y tế"
            
            specialty_data = pkg.get("specialtyData", {}) or {}
            specialty_name = specialty_data.get("name", "Đa khoa") if isinstance(specialty_data, dict) else "Đa khoa"
            
            included = pkg.get("includedServices", "") or ""
            requirements = pkg.get("requirements", "") or ""
            notes = pkg.get("notes", "") or ""
            
            pkg_text = f"""Gói khám bệnh: {name} (Phân loại: {category}).
Giá trọn gói: {price_str}.
Cơ sở triển khai: {facility_name}. Chuyên khoa liên quan: {specialty_name}.
Đối tượng: {gender_vn}, Độ tuổi phù hợp: {age_range}. Thời gian khám dự kiến: {duration} phút.
Mô tả gói khám: {desc}
Danh mục dịch vụ bao gồm: {included if included else 'Khám lâm sàng, xét nghiệm tổng quát và chẩn đoán cận lâm sàng.'}
Yêu cầu chuẩn bị trước khi khám: {requirements if requirements else 'Nên nhịn ăn sáng trước khi làm các xét nghiệm đường huyết/máu.'}
Lưu ý: {notes if notes else 'Nên đặt lịch trước ít nhất 1 ngày để cơ sở y tế chuẩn bị chu đáo.'}
Từ khóa: gói khám {name}, khám sức khỏe {name}, tầm soát {name}, giá gói {name}, {category}, gói khám tổng quát"""

            pkg_copy = {k: v for k, v in pkg.items() if k not in ["image", "logo"]}
            documents.append(Document(
                page_content=pkg_text.strip(),
                metadata={
                    "type": "package",
                    "package_name": name,
                    "category": category,
                    "price": price_str,
                    "facility_name": facility_name,
                    "specialty_name": specialty_name,
                    "data": pkg_copy
                }
            ))
            logger.info(f"Added package doc: {name} ({price_str})")
        return documents

    def _process_facilities(self, facilities: List[Dict]) -> List[Document]:
        """Xử lý danh sách cơ sở y tế / bệnh viện đối tác."""
        documents = []
        for fac in facilities:
            name = (fac.get("name") or "").strip()
            if not name:
                continue
            desc = fac.get("description", "") or ""
            address = fac.get("address", "") or ""
            phone = fac.get("phone", "") or ""
            email = fac.get("email", "") or ""
            website = fac.get("website", "") or ""
            fac_type = fac.get("type", "Bệnh viện") or "Bệnh viện"
            level = fac.get("level", "Tuyến đầu") or "Tuyến đầu"
            working_hours = fac.get("workingHours", "07:30 - 17:00 (Thứ 2 - Thứ 7)") or "07:30 - 17:00"
            services = fac.get("services", "") or ""
            specialties = fac.get("specialties", "") or ""
            rating = fac.get("rating", 5.0)
            reviews = fac.get("totalReviews", 0)

            fac_text = f"""Bệnh viện / Cơ sở y tế: {name} (Phân loại: {fac_type}, Hạng: {level}).
Địa chỉ: {address}.
Điện thoại hotline: {phone}. Email: {email}. Website: {website}.
Thời gian làm việc: {working_hours}.
Đánh giá chất lượng: {rating}/5 sao ({reviews} lượt đánh giá).
Chuyên khoa mũi nhọn: {specialties}.
Dịch vụ kỹ thuật cung cấp: {services}.
Giới thiệu khái quát: {desc}
Từ khóa: bệnh viện {name}, cơ sở y tế {name}, phòng khám {name}, địa chỉ {name}, liên hệ {name}, cấp cứu {name}"""

            fac_copy = {k: v for k, v in fac.items() if k not in ["logo", "images", "image"]}
            documents.append(Document(
                page_content=fac_text.strip(),
                metadata={
                    "type": "facility",
                    "facility_name": name,
                    "address": address,
                    "phone": phone,
                    "data": fac_copy
                }
            ))
            logger.info(f"Added facility doc: {name} ({address})")
        return documents

    def _process_medicines(self, medicines: List[Dict]) -> List[Document]:
        """Xử lý danh mục thuốc và dược phẩm."""
        documents = []
        for med in medicines:
            name = (med.get("name") or "").strip()
            if not name:
                continue
            unit = med.get("unit", "Hộp") or "Hộp"
            desc = med.get("description", "") or ""
            price = med.get("price", 0)
            try:
                price_str = f"{float(price):,.0f} VNĐ/{unit}" if price else "Niêm yết tại quầy"
            except:
                price_str = str(price)

            med_text = f"""Dược phẩm / Thuốc: {name}. Đơn vị tính: {unit}.
Giá niêm yết: {price_str}.
Công dụng chính và chỉ định: {desc}
Lưu ý y khoa quan trọng: Bệnh nhân chỉ sử dụng thuốc theo đúng liều lượng chỉ định của bác sĩ chuyên khoa, không tự ý mua uống.
Từ khóa: thuốc {name}, tác dụng thuốc {name}, giá {name}, đơn vị {name}"""

            med_copy = {k: v for k, v in med.items()}
            documents.append(Document(
                page_content=med_text.strip(),
                metadata={
                    "type": "medicine",
                    "medicine_name": name,
                    "price": price_str,
                    "data": med_copy
                }
            ))
            logger.info(f"Added medicine doc: {name}")
        return documents

    def as_retriever(self, **kwargs):
        """Trả về retriever từ vector store."""
        if self.vectorstore is None:
            logger.warning("⚠️ Vector store not initialized")
            return None
        return self.vectorstore.as_retriever(**kwargs)

    def hybrid_search_with_rerank(self, query: str, k: int = 5, target_province: Optional[str] = None, target_district: Optional[str] = None) -> List[Document]:
        """
        Tìm kiếm lai (Hybrid Search) kết hợp Dense Vector Search (Qdrant/FAISS) 
        và Sparse Keyword Search (BM25), sau đó chạy một bước Rerank để tăng trọng số
        các tài liệu có từ khóa trùng khớp trực tiếp hoặc cùng tỉnh thành/quận huyện với người dùng.
        """
        if self.vectorstore is None:
            logger.warning("⚠️ Vector store not initialized. Cannot perform search.")
            return []

        # 1. Lấy kết quả từ Dense Vector Search (Lấy k * 2 candidates)
        dense_docs = []
        try:
            dense_docs = self.vectorstore.similarity_search(query, k=k*2)
        except Exception as e:
            logger.error(f"❌ Dense search failed: {e}")

        # 2. Lấy kết quả từ Sparse Search (BM25) (Lấy k * 2 candidates)
        sparse_docs = []
        if self.bm25:
            try:
                tokenized_query = tokenize_vietnamese(query)
                scores = self.bm25.get_scores(tokenized_query)
                # Lấy top indices
                top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k*2]
                sparse_docs = [self.raw_documents[idx] for idx in top_indices if scores[idx] > 0]
            except Exception as e:
                logger.error(f"❌ BM25 search failed: {e}")

        # 3. Reciprocal Rank Fusion (RRF) để hợp nhất kết quả
        # RRF Score = Sum(1 / (60 + rank))
        rrf_scores = {} # (page_content, type) -> {"doc": Document, "score": float}
        
        def merge_results(doc_list):
            for rank, doc in enumerate(doc_list, 1):
                key = (doc.page_content, doc.metadata.get("type", ""))
                if key not in rrf_scores:
                    rrf_scores[key] = {"doc": doc, "score": 0.0}
                rrf_scores[key]["score"] += 1.0 / (60.0 + rank)

        merge_results(dense_docs)
        merge_results(sparse_docs)

        # Chuyển đổi và sắp xếp
        fused_docs = list(rrf_scores.values())
        fused_docs.sort(key=lambda item: item["score"], reverse=True)
        candidates = [item["doc"] for item in fused_docs[:k*2]]

        # 4. Reranker (Tăng điểm cho các tài liệu khớp chính xác thông tin quan trọng)
        query_lower = query.lower()
        reranked_docs = []

        for doc in candidates:
            doc_type = doc.metadata.get("type", "")
            boost_score = 0.0

            # Lấy toàn bộ thông tin địa chỉ của tài liệu
            doc_addr_full = (
                str(doc.metadata.get("province", "")) + " " + 
                str(doc.metadata.get("address", "")) + " " + 
                str(doc.metadata.get("addressClinic", "")) + " " + 
                str(doc.metadata.get("data", {}).get("province", "")) + " " + 
                str(doc.metadata.get("data", {}).get("addressClinic", "")) + " " + 
                str(doc.metadata.get("data", {}).get("address", ""))
            ).lower()

            # Boost mạnh nếu cơ sở/bác sĩ cùng tỉnh thành với người dùng (+4.0)
            if target_province:
                t_prov = target_province.lower()
                prov_match = False
                if any(x in t_prov for x in ["hồ chí minh", "sài gòn", "hcm"]):
                    prov_match = any(alias in doc_addr_full for alias in ["hồ chí minh", "ho chi minh", "hcm", "tp. hcm", "sài gòn"])
                else:
                    prov_match = t_prov in doc_addr_full

                if prov_match:
                    boost_score += 4.0

            # Boost vượt trội nếu khớp đúng Quận / Huyện (+3.5)
            if target_district:
                t_dist = target_district.lower()
                dist_match = False
                if t_dist.startswith("quận "):
                    q_num = t_dist.replace("quận ", "").strip()
                    dist_match = any(pat in doc_addr_full for pat in [t_dist, f"q.{q_num}", f"q{q_num}", f"district {q_num}"])
                else:
                    dist_match = t_dist in doc_addr_full or t_dist in doc.page_content.lower()

                if dist_match:
                    boost_score += 3.5

            # Boost bác sĩ nếu khớp tên
            if doc_type == "doctor":
                doc_data = doc.metadata.get("data", {})
                first_name = doc_data.get("firstName", "").lower()
                last_name = doc_data.get("lastName", "").lower()
                full_name = f"{first_name} {last_name}".strip()
                if (first_name and first_name in query_lower) or (last_name and last_name in query_lower):
                    boost_score += 2.0
                if full_name and full_name in query_lower:
                    boost_score += 3.0

            # Boost chuyên khoa nếu khớp tên
            elif doc_type in ["specialty", "specialty_detail"]:
                spec_name = doc.metadata.get("specialty_name", "").lower()
                if spec_name and spec_name in query_lower:
                    boost_score += 2.5

            # Boost phòng khám/bệnh viện nếu khớp tên
            elif doc_type in ["clinic", "clinic_detail"]:
                clinic_name = doc.metadata.get("clinic_name", "").lower()
                if clinic_name and clinic_name in query_lower:
                    boost_score += 1.5

            # Boost gói khám nếu query hỏi về gói khám hoặc khớp tên gói
            elif doc_type == "package":
                pkg_name = doc.metadata.get("package_name", "").lower()
                pkg_category = doc.metadata.get("category", "").lower()
                if any(w in query_lower for w in ["gói", "goi", "tổng quát", "tầm soát", "combo"]):
                    boost_score += 2.0
                if pkg_name and (pkg_name in query_lower or any(w in query_lower for w in pkg_name.split())):
                    boost_score += 2.5
                if pkg_category and pkg_category in query_lower:
                    boost_score += 1.5

            # Boost cơ sở y tế / bệnh viện đối tác
            elif doc_type == "facility":
                fac_name = doc.metadata.get("facility_name", "").lower()
                if any(w in query_lower for w in ["bệnh viện", "benh vien", "cơ sở", "trung tâm", "tuyến"]):
                    boost_score += 1.5
                if fac_name and (fac_name in query_lower or any(w in query_lower for w in fac_name.split())):
                    boost_score += 2.5

            # Boost thuốc nếu hỏi về thuốc
            elif doc_type == "medicine":
                med_name = doc.metadata.get("medicine_name", "").lower()
                if any(w in query_lower for w in ["thuốc", "thuoc", "uống", "viên", "liều"]):
                    boost_score += 1.5
                if med_name and med_name in query_lower:
                    boost_score += 3.0

            reranked_docs.append((doc, boost_score))

        # Sắp xếp theo boost score trước, sau đó là thứ tự RRF gốc
        reranked_docs.sort(key=lambda pair: pair[1], reverse=True)
        
        # Chọn ra top k kết quả cuối cùng
        final_docs = [doc for doc, boost in reranked_docs[:k]]
        return final_docs

    def search_doctors_by_specialty(self, specialty_query: str, k: int = 5) -> List[Dict]:
        """Tìm kiếm bác sĩ theo chuyên khoa cụ thể."""
        if not self.vectorstore:
            return []
            
        try:
            # Tìm kiếm documents liên quan
            docs = self.vectorstore.similarity_search(
                f"bác sĩ chuyên khoa {specialty_query}", 
                k=k*2  # Lấy nhiều hơn để filter
            )
            
            # Filter chỉ lấy doctors
            doctor_docs = [doc for doc in docs if doc.metadata.get("type") == "doctor"]
            return doctor_docs[:k]
            
        except Exception as e:
            logger.error(f"Error searching doctors by specialty: {e}")
            return []
