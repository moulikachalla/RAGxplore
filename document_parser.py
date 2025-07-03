# Standard library imports for file handling, path management, parsing, and error reporting
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import email
import traceback

# Attempt to import PyMuPDF for PDF parsing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    print("PyMuPDF is available and loaded successfully")
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available. PDF parsing will be limited.")

# DocumentParser class: Handles parsing of PDF, email (.eml), and text (.txt) files
class DocumentParser:
    def __init__(self, output_dir: str = "data/unstructured"):
        # Initialize output directory and document tracking
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.supported_extensions = {'.txt', '.pdf', '.eml'}
        self.processed_documents = []
        self.failed_documents = []

        print(f"\n DocumentParser initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   PyMuPDF available: {PYMUPDF_AVAILABLE}\n")

    # Parse PDF files using PyMuPDF
    def parse_pdf(self, file_path: Path) -> Optional[Dict]:
        if not PYMUPDF_AVAILABLE:
            print(f" PyMuPDF not available, skipping PDF: {file_path.name}")
            return None

        print(f"Parsing PDF: {file_path}")
        try:
            with fitz.open(str(file_path)) as doc:
                print(f"PDF has {len(doc)} pages")
                # Extract text from all pages
                content = "".join([page.get_text() for page in doc])
                print(f"Successfully extracted {len(content)} characters")

                # Return structured document dictionary
                return {
                    "id": str(uuid.uuid4()),
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "content": content.strip(),
                    "metadata": {
                        "file_type": "pdf",
                        "document_type": "pdf_document",
                        "file_size": file_path.stat().st_size,
                        "pages": len(doc),
                        "created_at": datetime.now().isoformat(),
                        "characters": len(content.strip())
                    }
                }
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            print(traceback.format_exc())
            return None

    # Parse email (.eml) files
    def parse_email(self, file_path: Path) -> Optional[Dict]:
        print(f"Parsing email: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()

            msg = email.message_from_string(email_content)

            # Extract email headers
            subject = msg.get('Subject', 'No Subject')
            from_addr = msg.get('From', 'Unknown Sender')
            to_addr = msg.get('To', 'Unknown Recipient')
            date = msg.get('Date', 'Unknown Date')

            # Extract email body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode('utf-8', errors='ignore')
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')

            print(f" Extracted {len(body)} characters from email body")

            # Combine header + body into a single content string
            formatted_content = f"Subject: {subject}\n\nFrom: {from_addr}\nTo: {to_addr}\nDate: {date}\n\n{body}"

            return {
                "id": str(uuid.uuid4()),
                "filename": file_path.name,
                "filepath": str(file_path),
                "content": formatted_content.strip(),
                "metadata": {
                    "file_type": "email",
                    "document_type": "email",
                    "file_size": file_path.stat().st_size,
                    "subject": subject,
                    "from": from_addr,
                    "to": to_addr,
                    "date": date,
                    "created_at": datetime.now().isoformat(),
                    "characters": len(formatted_content.strip())
                }
            }
        except Exception as e:
            print(f"Error parsing email {file_path}: {e}")
            return None

    # Parse plain text (.txt) files
    def parse_text_file(self, file_path: Path) -> Optional[Dict]:
        print(f"Parsing text file: {file_path}")
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore').strip()
            print(f" Read {len(content)} characters from text file")
            return {
                "id": str(uuid.uuid4()),
                "filename": file_path.name,
                "filepath": str(file_path),
                "content": content,
                "metadata": {
                    "file_type": "text",
                    "document_type": "text_document",
                    "file_size": file_path.stat().st_size,
                    "created_at": datetime.now().isoformat(),
                    "characters": len(content)
                }
            }
        except Exception as e:
            print(f"Error parsing text file {file_path}: {e}")
            return None

    # Process a single file based on its extension
    def process_file(self, file_path: Path) -> bool:
        print(f" Processing: {file_path.name}\n")
        extension = file_path.suffix.lower()

        # Skip unsupported file types
        if extension not in self.supported_extensions:
            print(f"Skipping unsupported file type: {file_path.name}")
            return False

        try:
            # Use a map to select the right parsing method
            parser_map = {
                '.pdf': self.parse_pdf,
                '.eml': self.parse_email,
                '.txt': self.parse_text_file
            }
            document = parser_map.get(extension, lambda x: None)(file_path)

            if document:
                self.processed_documents.append(document)
                print(f"      Successfully parsed: {file_path.name}")
                print(f"      - Document ID: {document['id']}")
                print(f"      - Content length: {len(document['content'])} characters")
                return True
            else:
                self.failed_documents.append(str(file_path))
                print(f"Failed to parse: {file_path.name}")
                return False
        except Exception as e:
            print(f"Unexpected error processing {file_path.name}: {e}")
            self.failed_documents.append(str(file_path))
            return False

    # Process all files in a directory
    def process_directory(self, input_dir: str) -> None:
        input_path = Path(input_dir)
        print(f"Checking input directory: {input_dir}")

        if not input_path.exists():
            print(f" Input directory does not exist: {input_dir}")
            return

        # List all files in directory
        files = [f for f in input_path.iterdir() if f.is_file()]
        print(f"Found {len(files)} files\n")

        for file_path in files:
            self.process_file(file_path)

    # Save parsed documents to JSONL file
    def save_to_jsonl(self, output_filename: str = "parsed.jsonl") -> None:
        if not self.processed_documents:
            print("No documents to save")
            return

        output_path = self.output_dir / output_filename
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in self.processed_documents:
                    json.dump(doc, f, ensure_ascii=False)
                    f.write('\n')
            print(f"Saved {len(self.processed_documents)} documents to {output_path}")
        except Exception as e:
            print(f"Error saving to JSONL: {e}")
            print(traceback.format_exc())

    # Print a summary of all parsed and failed documents
    def print_summary(self) -> None:
        total_docs = len(self.processed_documents)
        print(f"\nTotal processed documents: {total_docs}\n")

        if total_docs > 0:
            from collections import Counter
            total_size = sum(doc.get('metadata', {}).get('file_size', 0) for doc in self.processed_documents)
            total_chars = sum(doc.get('metadata', {}).get('characters', len(doc.get('content', ''))) for doc in self.processed_documents)
            doc_types = Counter(doc.get('metadata', {}).get('document_type', 'unknown') for doc in self.processed_documents)

            print(f" Summary:")
            print(f"   Total size: {total_size:,} bytes")
            print(f"   Total characters: {total_chars:,}")
            print(f"   Document types:")
            for doc_type, count in doc_types.items():
                print(f"      - {doc_type}: {count} files")

            print(f"\nSample Preview:")
            for i, doc in enumerate(self.processed_documents[:3], 1):
                print(f"--- DOCUMENT {i} ---")
                print(f"File: {doc['filepath']}")
                print(f"Content (first 200 chars): {doc['content'][:200]}\n")

        if self.failed_documents:
            print(f"Failed to process {len(self.failed_documents)} files:")
            for path in self.failed_documents:
                print(f"   - {path}")

# Entry point to run the parser
def main():
    print("\n Starting Document Parser\n")
    parser = DocumentParser()
    parser.process_directory("data/unstructured")
    parser.save_to_jsonl()
    parser.print_summary()

if __name__ == "__main__":
    main()
