import os
import time
import threading
from typing import List, Optional, Union
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
class DocumentIndexer:
    def __init__(
        self, 
        index_path: str = "faiss_index", 
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        cache_folder: Optional[str] = "D:/我的文件/v_llm_live2d_tts/",
        save_document_interval: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化文档索引器
        
        :param index_path: FAISS索引保存路径
        :param embedding_model: 嵌入模型名称
        :param cache_folder: 模型缓存路径
        :param save_document_interval: 保存文档的间隔
        :param chunk_size: 文本分块大小
        :param chunk_overlap: 文本分块重叠大小
        """
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            cache_folder=cache_folder
        )
        
        self.index_path = index_path
        self.save_document_interval = save_document_interval
        
        # 加载或创建向量存储
        self.vectorstore = self._load_or_create_vectorstore()
        
        self.temporal_document = []
        # 启动保存线程
        self._start_periodic_save()
    
    def _load_or_create_vectorstore(self) -> FAISS:
        """
        加载已存在的索引，如果不存在则创建新的
        """
        try:
            # 尝试加载现有索引
            print(f"尝试加载现有索引: {self.index_path}")
            return FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"加载索引时出现错误: {e}")
            # 如果加载失败，创建空的向量存储
            print("索引不存在，创建新的向量存储")
            return FAISS.from_texts(["初始文本"], self.embeddings)
    
    def _start_periodic_save(self):
        """
        启动定时保存线程
        """
        def save_periodically():
            while True:
                time.sleep(10)
                if len(self.temporal_document) >= self.save_document_interval:
                    self._process_and_index_documents()
        
        save_thread = threading.Thread(target=save_periodically, daemon=True)
        save_thread.start()
    
    def add_text(self, text: str, metadata: Optional[dict] = None):
        """
        添加文本到临时文档列表
        
        :param text: 文本内容
        :param metadata: 文本的元数据
        """
        doc = Document(
            page_content=text, 
            metadata=metadata or {}
        )
        self.temporal_document.append(doc)

    def _process_and_index_documents(self):
        """
        处理并索引临时文档
        """
        if not self.temporal_document:
            return

        # try:
        # 分割文档
        split_docs = []
        # 使用文本分割器对文档进行分块
        chunks = self.text_splitter.split_documents(self.temporal_document)
        split_docs.extend(chunks)

        # 如果有分割后的文档
        if split_docs:
            # 使用FAISS向量存储添加文档
            if self.vectorstore is None:
                # 如果向量存储不存在，创建新的
                self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            else:
                # 如果向量存储已存在，添加新文档
                self.vectorstore.add_documents(split_docs)

            # 保存索引
            self.save_index()

            # 清空临时文档列表
            self.temporal_document.clear()
            
            print(f"成功处理并索引 {len(split_docs)} 个文档块")
        
        # except Exception as e:
        #     print(f"处理文档时发生错误: {e}")

    def save_index(self):
        """
        保存当前索引
        """
        try:
            if self.vectorstore:
                self.vectorstore.save_local(self.index_path)
                print(f"索引已保存到 {self.index_path}")
        except Exception as e:
            print(f"保存索引时出现错误: {e}")

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        根据查询搜索相关文档
        
        :param query: 查询文本
        :param k: 返回的文档数量
        :return: 相关文档列表
        """
        if self.vectorstore is None:
            return []
        
        return self.vectorstore.similarity_search(query, k=k)

if __name__ == "__main__":
    # 创建文档索引器
    indexer = DocumentIndexer(
        index_path="memory", 
        embedding_model="BAAI/bge-small-zh-v1.5"
    )


    # 添加文档方式2：添加长文本
    long_text = """
    机器学习是人工智能的一个重要分支。它通过算法和统计模型，使计算机系统能够在没有明确编程的情况下改进性能。
    机器学习算法主要分为三类：
    1. 监督学习：使用标记数据训练模型
    2. 非监督学习：从未标记的数据中发现模式
    3. 强化学习：通过与环境交互来学习
    """
    # indexer.add_text("董玉博是大学生")
    # indexer.add_text("机器学习是人工智能的一个重要分支")
    # indexer.add_text("深度学习是机器学习的一种方法")
    # indexer.add_text("自然语言处理是人工智能的一个重要领域")
    # indexer.add_text("计算机视觉是人工智能的一个重要领域")
    # indexer.add_text(long_text)

    # 等待索引处理（因为有后台线程）
    import time
    time.sleep(15)  # 等待后台线程处理文档

    # 搜索相关文档
    query = "董玉博"
    results = indexer.search(query, k=1)

    print(f"查询 '{query}' 的搜索结果:")
    for i, doc in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")

    # 再次搜索，测试不同查询
    query2 = "机器学习"
    results2 = indexer.search(query2, k=2)

    print(f"\n\n查询 '{query2}' 的搜索结果:")
    for i, doc in enumerate(results2, 1):
        print(f"\n结果 {i}:")
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
