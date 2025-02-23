"""
文档处理模块
处理各种类型的文档并提取信息
"""

import os
from datetime import datetime
import docx
import pandas as pd
import PyPDF2
from PIL import Image
import pytesseract
import json
import requests
from bs4 import BeautifulSoup
from typing import Dict
from ..config.settings import SUPPORTED_EXTENSIONS

class DocumentProcessor:
    """文档处理器类，用于处理各种类型的文档并提取信息。"""
    
    def __init__(self):
        """初始化文档处理器"""
        self.supported_extensions = SUPPORTED_EXTENSIONS
    
    def validate_file(self, file) -> bool:
        """验证文件格式是否支持"""
        if not file:
            return False
        ext = os.path.splitext(file.name)[1].lower()
        return any(ext in exts for exts in self.supported_extensions.values())
    
    def extract_text_from_docx(self, file) -> str:
        """从Word文档中提取文本"""
        doc = docx.Document(file)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    def extract_text_from_pdf(self, file) -> str:
        """从PDF文件中提取文本"""
        pdf_reader = PyPDF2.PdfReader(file)
        return '\n'.join([page.extract_text() for page in pdf_reader.pages])
    
    def extract_text_from_image(self, file) -> str:
        """从图片中提取文本"""
        image = Image.open(file)
        return pytesseract.image_to_string(image, lang='chi_sim+eng')
    
    def extract_data_from_excel(self, file) -> Dict:
        """从Excel文件中提取数据"""
        df = pd.read_excel(file)
        return {
            'headers': df.columns.tolist(),
            'data': df.values.tolist(),
            'summary': df.describe().to_dict()
        }
    
    def extract_from_url(self, url: str) -> str:
        """从URL中提取内容"""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    
    def process_file(self, file) -> Dict:
        """处理上传的文件并提取信息"""
        try:
            file_ext = os.path.splitext(file.name)[1].lower()
            content = ''
            
            if file_ext in self.supported_extensions['document']:
                content = self.extract_text_from_docx(file)
            elif file_ext in self.supported_extensions['pdf']:
                content = self.extract_text_from_pdf(file)
            elif file_ext in self.supported_extensions['text']:
                content = file.getvalue().decode('utf-8')
            elif file_ext in self.supported_extensions['image']:
                content = self.extract_text_from_image(file)
            elif file_ext in self.supported_extensions['spreadsheet']:
                content = json.dumps(self.extract_data_from_excel(file))
            
            return {
                'filename': file.name,
                'content': content,
                'type': file_ext,
                'size': file.size,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'filename': file.name,
                'error': str(e),
                'type': file_ext,
                'size': file.size,
                'timestamp': datetime.now().isoformat()
            } 