import os, sys,re,json
from pathlib import Path
import fitz,pdfplumber
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from typing import Optional,Dict,List

doc_path = Path(__file__).resolve().parent.parent / "Data"
pdf_paths = doc_path / "ISO-7001-2023.pdf"

#Detect Images:
def detect_images(pdf_path, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_name = f"page_{page_num+1}.png"
        image_path = os.path.join(output_dir, image_name)
        pix.save(image_path)

        images.append({
            "page_num": page_num + 1,
            "image_path": image_path
        })
        # blocks = page.get_text("dict")["blocks"]

        # img_index = 0
        # for block in blocks:
        #     if block["type"] == 1 and "xref" in block:  # image block
        #         bbox = block["bbox"]
        #         pix = page.get_pixmap(clip = fitz.Rect(bbox))
        #         if pix.n > 4:
        #             pix = fitz.Pixmap(fitz.csRGB, pix)

        #         image_name = f"page_{page_num+1}_Img_{img_index}.png"
        #         image_path = os.path.join(output_dir, image_name)
        #         pix.save(image_path)

        #         images.append({
        #             "page_num": page_num + 1,
        #             "bbox": bbox,
        #             "image_path": image_path
        #         })

        #         img_index += 1
                #   pix = None
    doc.close()
    return images
detect_images(pdf_paths)


#Detect Headings:
def detect_heading(line:str,font_info:Optional[Dict]=None)->str:
    line = line.strip()
    if not line:
        return None
    
    match = re.match(r'^(\d+(?:\.\d+){0,3})\s+(.+)$', line)
    if match:
        prefix = match.group(1)
        dot_count = prefix.count(".")

        if dot_count ==0:
           classification = "heading"
        elif dot_count==1:
            classification = "sub_heading"
        elif dot_count ==2:
            classification = "sub_sub_heading"
        else:
            classification = None

        if font_info:
            font_name = font_info.get("font","").strip.lower()
            font_size = font_info.get("size",0)
            if "bold" or "italic" in font_info or font_size>=12:
                return classification
            else:
                return None
        
        return classification
        
    #Non numbered but bold
    if font_info:
        font_name = font_info.get("font", "").lower()
        font_size = font_info.get("size", 0)
        if "bold" in font_name and font_size >= 10:
            return "sub_heading"
        
    #Number headings
    only_number = re.match(r'^(\d+(?:\.\d+){0,3})$', line)
    if only_number:
        prefix = only_number.group(1)
        dot_count = prefix.count(".")

        if dot_count == 0:
            return "heading"
        elif dot_count == 1:
            return "sub_heading"
        
    return None

#Extraction of text:
def extract_text(pdf_path):
    doc =fitz.open(pdf_path)
    structured_data = []
    current_section = None
    current_sub_section = None
    current_sub_sub_section = None

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_dict = page.get_text("dict")
        page_lines = []
        
        for blocks in page_dict.get("blocks",[]):
            for lines in blocks.get("lines",[]):
                line_text = "\n".join([span.get('text') for span in lines.get("spans",[])])
                # print(repr(line_text))
                if not line_text:
                    continue
                
                if (
                        re.match(r"^ISO\s+\d{4,5}:\d{4}\(E\)$", line_text)
                        or re.search(r"© ISO \d{4} – All rights reserved", line_text)
                   
                    ):
                        continue
                page_lines.append(line_text)     
        # print(page_lines)

        #Paragraph initiation
        current_para = ""
        for line in page_lines:
            line = line.replace('\ufeff', '').replace('\x08', '').replace('\xa0','').strip()
            line = re.sub(r'\s+', ' ', line).strip()
            if not line:
                if current_para:
                    structured_data.append({
                        "Document_name": pdf_path,
                                "page_num": page_num + 1,
                                "heading": current_section,
                                "sub_heading": current_sub_section,
                                "sub_sub_heading": current_sub_sub_section,
                                "paragraph": current_para.strip()
                    })

                    current_para = ""
                continue

            #Detect heading
            classification = detect_heading(line)
            if classification in ["heading","sub_heading","sub_sub_heading"]:
                if current_para:
                    structured_data.append({
                        "Document_name": pdf_path,
                                "page_num": page_num + 1,
                                "heading": current_section,
                                "sub_heading": current_sub_section,
                                "sub_sub_heading": current_sub_sub_section,
                                "paragraph": current_para.strip()
                    })

                    current_para = ""

            # Update trackers
                if classification == "heading":
                    current_section = line
                    current_sub_section = None
                    current_sub_sub_section = None
                elif classification == "sub_heading":
                    current_sub_section = line
                    current_sub_sub_section = None
                elif classification == "sub_sub_heading":
                    current_sub_sub_section = line
            

            if re.match(r"^([a-z]\)|\d+\.|\-|•)\s+.+", line):
                if current_para:
                    current_para += " " + line
                else:
                    current_para = line
            else:
                if current_para:
                    current_para += " " + line
                else:
                    current_para = line

            # Finalize last paragraph on page
        if current_para:
            structured_data.append({
                "Document_name":pdf_path,
                "page_num": page_num + 1,
                "heading": current_section,
                "sub_heading": current_sub_section,
                "sub_sub_heading": current_sub_sub_section,
                "paragraph": current_para.strip()
            })
    doc.close()
    return {"Paragraphs":structured_data}

def attach_images_to_paragraphs(structured_data, images):
    """
    Attach images to all paragraphs under the same heading context
    until the next heading appears.
    """
    # Group paragraphs by page
    page_paragraphs = {}
    for idx, para in enumerate(structured_data):
        page_paragraphs.setdefault(para["page_num"], []).append(idx)

    for img in images:
        pg_num = img["page_num"]
        if pg_num not in page_paragraphs:
            continue

        # Attach image to all paragraphs on the page (context until next heading)
        for idx in page_paragraphs[pg_num]:
            if "images" not in structured_data[idx]:
                structured_data[idx]["images"] = []
            structured_data[idx]["images"].append(img["image_path"])

    return structured_data

pdf_path = pdf_paths
output_dir = "images"

# 1.Extract text
text_data = extract_text(pdf_path)
paragraphs = text_data["Paragraphs"]

# 2. Extract images
images = detect_images(pdf_path, output_dir=output_dir)

# 3. Attach images contextually
final_data = attach_images_to_paragraphs(paragraphs, images)

