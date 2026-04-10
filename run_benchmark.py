import json
import glob
import os
from pathlib import Path

from src.models import Document
from src.chunking import RecursiveChunker
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

def main():
    # Đọc tham số từ .env
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL")

    print("Khởi tạo OpenAIEmbedder (llama-nemotron-embed-1b-v2)...")
    from src.embeddings import OpenAIEmbedder
    embedder = OpenAIEmbedder()
    
    # 1. Đọc tất cả các file markdown rapper
    data_dir = Path("data/raw_data")
    md_files = glob.glob(str(data_dir / "*.md"))
    print(f"Đã tìm thấy {len(md_files)} file tài liệu.")

    # 2. Xử lý chia nhỏ văn bản (Chunking) bằng RecursiveChunker
    chunker = RecursiveChunker(chunk_size=150)   
    docs_to_store = []
    chunk_id_counter = 0

    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        filename = os.path.basename(file_path)
        rapper_name = filename.replace(".md", "")
        
        # Split into chunks
        chunks = chunker.chunk(content)
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            chunk_id_counter += 1
            doc = Document(
                id=f"chunk_{chunk_id_counter}",
                content=chunk,
                metadata={
                    "rapper": rapper_name,
                    "source": filename
                }
            )
            docs_to_store.append(doc)

    print(f"Đã tạo {len(docs_to_store)} chunks từ tài liệu.")

    # 3. Đưa vào Embedding Store
    print("Đang embedding và lưu vào store...")
    store = EmbeddingStore(collection_name="rappers_info", embedding_fn=embedder)
    store.add_documents(docs_to_store)
    print("Store size:", store.get_collection_size())

    # 4. Đọc benchmark.json và đánh giá
    with open("benchmark.json", "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # Hàm gọi LLM thật (gpt-4o-mini)
    def openai_llm_fn(prompt: str) -> str:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the given context to answer the question. If the answer is not in the context, say 'I cannot answer this based on the retrieved context.'"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    agent = KnowledgeBaseAgent(store=store, llm_fn=openai_llm_fn)

    print("\n" + "="*50)
    print("BẮT ĐẦU CHẠY BENCHMARK QUERIES")
    print("="*50)

    for b in benchmarks:
        q_id = b["id"]
        question = b["question"]
        expected = b["answer"]
        
        print(f"\n[Câu hỏi {q_id}] {question}")
        print(f"--- Câu trả lời kỳ vọng (Gold Answer):")
        print(f"    {expected[:150]}..." if len(expected) > 150 else f"    {expected}")
        
        # Tìm hiểu xem Agent retrieve được context nào
        retrieved_results = store.search(question, top_k=3)
        print(f"--- Top 3 Retrieved Chunks:")
        for idx, r in enumerate(retrieved_results):
            score = round(r["score"], 4)
            rapper = r["metadata"]["rapper"]
            chunk_snippet = r["content"].replace('\n', ' ')[:100]
            print(f"    {idx+1}. Thẩm định: {score} | Nguồn: {rapper}.md | Trích: '{chunk_snippet}...'")
        
        # Lấy câu trả lời từ RAG Agent
        agent_answer = agent.answer(question, top_k=3)
        print(f"\n--- RAG Agent Answer (GPT-4o-mini):")
        print(f"    {agent_answer}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
