from langchain.schema import Document
from langchain.vectorstores import FAISS
import faiss
import logging
from typing import Dict, Optional, List

logger = logging.getLogger("enhanced_medical_chatbot")

class EnhancedVectorStore:
    def __init__(self, embeddings, dimension: int = 384):
        self.embeddings = embeddings
        self.dimension = dimension
        self.vectorstore: Optional[FAISS] = None
        self.specialty_mapping = {}  # specialty_id -> specialty_info
        self.doctor_specialty_mapping = {}  # doctor_id -> specialty_info
    
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

        # Xây dựng FAISS vector store
        if documents:
            try:
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
                self.vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
                logger.info(f"✅ Built vector store with {len(documents)} documents")

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
            note = doctor.get("note", "")
            price = doctor.get("price", "")
            payment = doctor.get("payment", "")
            province = doctor.get("province", "")
            
            if name:
                # Tạo text phong phú cho doctor với liên kết chuyên khoa
                doctor_text = f"""Bác sĩ {name}
                                    Chuyên khoa: {specialty_name}
                                    Phòng khám: {clinic_name}
                                    Khu vực: {province}
                                    Giá khám: {price}
                                    Thanh toán: {payment}
                                    Ghi chú: {note[:200]}"""

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
                if name or description:
                    text = f"Bệnh viện {name}\n{description}".strip()
                    data_copy = {k: v for k, v in clinic_data.items() if k != "image"}
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "type": "clinic", 
                            "clinic_name": name,
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
                    text = f"Bệnh viện {clinic_name}\n{content}".strip()
                    md_copy = {k: v for k, v in clinic_md.items() if k != "image"}
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "type": "clinic_detail", 
                            "clinic_name": clinic_name,
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

    def as_retriever(self, **kwargs):
        """Trả về retriever từ vector store."""
        if self.vectorstore is None:
            logger.warning("⚠️ Vector store not initialized")
            return None
        return self.vectorstore.as_retriever(**kwargs)

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
