import os
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def extrair_texto_pdf(path_pdf):
    texto_final = ""

    # 🔍 Tentando extrair texto com pdfplumber (para PDFs normais)
    try:
        with pdfplumber.open(path_pdf) as pdf:
            for pagina in pdf.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_final += texto + "\n"
    except:
        pass

    # Se pdfplumber não conseguiu nada, fazemos OCR
    if len(texto_final.strip()) == 0:
        print(f"🔎 OCR ativado para: {path_pdf}")

        pdf = fitz.open(path_pdf)
        for pagina in pdf:
            pix = pagina.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            texto_final += pytesseract.image_to_string(img) + "\n"

    return texto_final


def processar_documentos():
    print("🔧 Iniciando processamento dos PDFs...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    arquivos = [
        "dados/politica_rh.pdf",
        "dados/manual_ti.pdf",
        "dados/onboarding.pdf"
    ]

    documentos = []

    for arquivo in arquivos:
        if not os.path.exists(arquivo):
            print(f"⚠️ Arquivo não encontrado: {arquivo}")
            continue

        print(f"📄 Lendo: {arquivo}")
        texto = extrair_texto_pdf(arquivo)

        if len(texto.strip()) == 0:
            print(f"❌ Não foi possível extrair texto de: {arquivo}")
            continue

        documentos.append(texto)

    # 🔪 Divisão em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    for doc in documentos:
        chunks.extend(splitter.split_text(doc))

    print(f"📚 Total de chunks criados: {len(chunks)}")

    # 🧠 Criar Chroma
    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="claro_base"
    )

    print("✅ Base vetorial atualizada com sucesso!")


if __name__ == "__main__":
    processar_documentos()
