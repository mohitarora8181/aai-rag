import os
import numpy as np
from transformers import pipeline
from pdf2image import convert_from_path
import easyocr
import cv2
from PIL import Image
import fitz
import streamlit as st
from huggingface_hub import snapshot_download
from langchain.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer, util
# import logging
from dotenv import load_dotenv

load_dotenv()

# logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", repo_type="model")
pipe = pipeline("object-detection", model="microsoft/table-transformer-detection")

ocr_reader = easyocr.Reader(['en'])
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def convert_pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

if 'group_block' not in st.session_state:
    st.session_state.group_block = None
if 'group_text' not in st.session_state:
    st.session_state.group_text = None

def extract_page_content(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_num, (image, page) in enumerate(zip(images, doc), 1):
        page_content = []
        detections = pipe(image)
        table_boxes = []

        for detection in detections:
            if 'box' in detection:
                box = detection['box']
                x0, y0, x1, y1 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
                table_boxes.append((y0, x0, y1, x1))  # Store as (y0, x0, y1, x1) for easier sorting

        image_cv2 = pil_to_cv2(image)
        img_np = np.array(image)
        img_np = img_np.astype(np.uint8)
        mask = np.ones(image_cv2.shape[:2], dtype=np.uint8) * 255  # Initial mask with white (255)

        for y0, x0, y1, x1 in table_boxes:
            mask[y0:y1, x0:x1] = 0
        if mask.shape[:2] != img_np.shape[:2]:
            mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

        img_masked = cv2.bitwise_and(img_np, img_np, mask=mask)

        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        ocr_results = ocr_reader.readtext(img_masked, paragraph=True)

        text_blocks = []
        for result in ocr_results:
            bbox, text = result[0], result[1]
            top_left_y = int(bbox[0][1])
            text_blocks.append((top_left_y, {"type": "plain_text", "text": text}))

        for y0, x0, y1, x1 in table_boxes:
            table_image = image.crop((x0, y0, x1, y1))
            table_image_cv2 = pil_to_cv2(table_image)
            table_ocr_results = ocr_reader.readtext(table_image_cv2)

            table_cells = [result[1] for result in table_ocr_results]

            text_blocks.append((y0, {
                "type": "table",
                "bounding_box": [x0, y0, x1, y1],
                "extracted_text": table_cells
            }))
        text_blocks.sort(key=lambda x: x[0])

        page_content = [block[1] for block in text_blocks]

        extracted_data.append({
            "page_number": page_num,
            "content": page_content
        })
        
    grouped_blocks = []

    for page in extracted_data:
        page_number = page['page_number']
        page_content = page['content']

        current_group = []
        for i, content in enumerate(page_content):
            current_group.append(content)
            is_last_block = (i == len(page_content) - 1)
            next_is_table = (not is_last_block and page_content[i + 1]["type"] == "table")
            current_is_table = content["type"] == "table"

            if current_is_table or next_is_table:
                continue

            if current_group:
                grouped_blocks.append({
                    "page_number": page_number,
                    "group_content": current_group,
                })
                current_group = []

        if current_group:
            grouped_blocks.append({
                "page_number": page_number,
                "group_content": current_group,
            })

    group_texts = []
    for block in grouped_blocks:
        block_text = ""
        for content in block["group_content"]:
            if content["type"] == "plain_text":
                block_text += content["text"] + " "
            elif content["type"] == "table":
                table_text = '; '.join([f"[{cell}]" for cell in content["extracted_text"]])
                block_text += f"Table Content: {table_text} "
        group_texts.append(block_text.strip())

    st.session_state.group_block = grouped_blocks
    st.session_state.group_text = group_texts

def answer_question(query):
    embeddings = embedding_model.encode(st.session_state.group_text, convert_to_tensor=True)
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()

    top_n = 10
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    retrieved_texts = []
    for idx in top_indices:
        block = st.session_state.group_block[idx]
        block_text = ""
        for content in block["group_content"]:
            if content["type"] == "plain_text":
                block_text += content["text"] + " "
            elif content["type"] == "table":
                table_text = '; '.join(content["extracted_text"])
                block_text += f"[Table]: {table_text} "
        retrieved_texts.append(f"Page {block['page_number']} - {block_text.strip()}")

    retrieved_text_final = "\n\n".join(retrieved_texts)

    prompt = f"""
    Use the following relevant document content to answer the question below:

    {retrieved_text_final}

    Question: {query}
    Helpful Answer:
    """
    
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        model_kwargs={"temperature": 0.1, "max_new_tokens": 300},
        huggingfacehub_api_token=os.environ["HF_TOKEN"]
    )

    response = llm.invoke(prompt,verbose=True)
    answer = response.split("Helpful Answer:")[1]
    return answer if(answer) else response

def main():
    st.title("RAG Q&A App")
    input_type = st.selectbox("Input Type", ["PDF", "Text", "DOCX", "TXT"])
    if input_type == "Text":
        input_data = st.text_input("Enter the text")
    elif input_type == 'PDF':
        input_data = st.file_uploader("Upload a PDF file", type=["pdf"])
    elif input_type == 'TXT':
        input_data = st.file_uploader("Upload a text file", type=['txt'])
    elif input_type == 'DOCX':
        input_data = st.file_uploader("Upload a DOCX file", type=[ 'docx', 'doc'])
    if st.button("Proceed"):
        input_path = os.path.join("./content/", input_data.name)
        with open(input_path, "wb") as f:
          f.write(input_data.getvalue())
          extract_page_content(input_path)
          
    if (st.session_state.group_block and st.session_state.group_text):
        query = st.text_input("Ask your question")
        if(query):
            answer = answer_question(query)
            if(answer):
                st.write(answer)

if __name__ == "__main__":
    main()
