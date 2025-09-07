from langchain.schema import Document
from langchain.vectorstores import FAISS
import faiss
import logging
from typing import Dict

logger = logging.getLogger("enhanced_medical_chatbot")

class EnhancedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        self.vectorstore = None
    
    def build_from_web_data(self, web_data: Dict):
        documents = []

        # Clinics
        for clinic in web_data.get("clinics", []):
            text = f"Bệnh viện {clinic.get('name', 'N/A')}: {clinic.get('description', '')}"
            doc = Document(page_content=text, metadata={"type": "clinic", "data": clinic})
            documents.append(doc)
            logger.debug(f"Added clinic doc: {text[:100]}...")

        # Specialties
        for specialty in web_data.get("specialties", []):
            text = f"Chuyên khoa {specialty.get('name', 'N/A')}: {specialty.get('description', '')}"
            doc = Document(page_content=text, metadata={"type": "specialty", "data": specialty})
            documents.append(doc)
            logger.debug(f"Added specialty doc: {text[:100]}...")

        # Handbooks
        for handbook in web_data.get("handbooks", []):
            for entry in handbook.get("handbookData", []):
                text = f"Cẩm nang y tế: {entry.get('title', 'N/A')}"
                entry_copy = {k: v for k, v in entry.items() if k != "image"}
                doc = Document(page_content=text, metadata={"type": "handbook", "data": entry_copy})
                documents.append(doc)
                logger.info(f"Added handbookData: {entry_copy.get('title', 'N/A')}")

            for md in handbook.get("handbookMarkdown", []):
                content = md.get("contentMarkdown", "")
                if content:
                    md_copy = {k: v for k, v in md.items() if k != "image"}
                    doc = Document(page_content=content, metadata={"type": "handbook", "data": md_copy})
                    documents.append(doc)
                    logger.info(f"Added handbookMarkdown (length={len(content)} chars)")

        # Doctors
        for doctor in web_data.get("doctors", []):
            name = f"{doctor.get('firstName', '')} {doctor.get('lastName', '')}".strip()
            text = f"Bác sĩ {name}: {doctor.get('specialty', '')}, {doctor.get('clinic', '')}"
            doc = Document(page_content=text, metadata={"type": "doctor", "data": doctor})
            documents.append(doc)
            logger.debug(f"Added doctor doc: {name}")

        # Build vector store
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
                # Log số lượng từng loại
                type_count = {}
                for d in documents:
                    t = d.metadata["type"]
                    type_count[t] = type_count.get(t, 0) + 1
                logger.info(f"Vector store breakdown: {type_count}")
            except Exception as e:
                logger.error(f"❌ Error building vector store: {e}")
                self.vectorstore = None
        else:
            logger.warning("⚠️ No documents to build vector store")
            self.vectorstore = None
    
    def as_retriever(self, **kwargs):
        if self.vectorstore is None:
            logger.warning("⚠️ Vector store not initialized")
            return None
        return self.vectorstore.as_retriever(**kwargs)
