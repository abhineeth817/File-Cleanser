<h1>🧹 File Cleanser</h1>
<h3>An AI-powered web app to automatically detect and cleanse sensitive data (PII) from multiple file formats.</h3>

<hr>

<h2>🚀 Overview</h2>
<p><b>File Cleanser</b> is a <b>React + Flask</b> application that removes <b>Personally Identifiable Information (PII)</b> from documents of various formats.  
It uses <b>NLP</b>, <b>OCR</b>, and <b>file parsing</b> modules to anonymize data while preserving document structure and readability.</p>

<hr>

<h2>🧠 Key Features</h2>
<ul>
  <li>🗂 <b>Multi-format Support:</b> TXT, DOCX, XLSX, CSV, PDF, PPTX, and Images</li>
  <li>🔍 <b>PII Detection:</b> Identifies names, emails, phone numbers, and other sensitive information</li>
  <li>🧩 <b>Dual Output Options:</b>
    <ul>
      <li>Cleansed <b>text file</b></li>
      <li>Cleansed file in its <b>original format</b> (except PPT)</li>
    </ul>
  </li>
  <li>🤖 <b>AI-Powered Extraction:</b>
    <ul>
      <li>Uses <b>Docling</b> for precise OCR-based text extraction from images</li>
      <li>Uses <b>fitz (PyMuPDF)</b>, <b>openpyxl</b>, and <b>docx</b> for other file types</li>
    </ul>
  </li>
  <li>🧼 <b>Anonymization:</b> Employs <b>Presidio Analyzer</b> and <b>Presidio Anonymizer</b> for PII redaction</li>
  <li>⚙️ <b>Flask API Backend</b> connects seamlessly with the <b>React frontend</b></li>
</ul>

<hr>

<h2>🧩 Tech Stack</h2>
<ul>
  <li><b>Frontend:</b> React</li>
  <li><b>Backend:</b> Python (Flask)</li>
  <li><b>AI/NLP:</b> spaCy, Presidio (Analyzer + Anonymizer)</li>
  <li><b>Text Extraction:</b> fitz (PyMuPDF), openpyxl, docx, Docling</li>
  <li><b>Other Tools:</b> threading, shutil, tempfile, regex</li>
</ul>

<hr>

<h2>⚙️ Setup and Run</h2>

<h3>1️⃣ Backend Setup (Python)</h3>
<pre><code>cd backend
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python backend34.py
</code></pre>

<h3>2️⃣ Frontend Setup (React)</h3>
<pre><code>cd frontend
npm install
npm start
</code></pre>

<p>Then visit <b>http://localhost:3000/</b> to access the app.</p>

<hr>

<h2>📂 Project Workflow</h2>
<ol>
  <li><b>File Upload:</b> User uploads a document via the React frontend</li>
  <li><b>Backend Processing:</b>
    <ul>
      <li>Flask handles file ingestion</li>
      <li>File-type-specific extractors retrieve raw text</li>
      <li>Docling performs OCR for image inputs</li>
      <li>Presidio detects and anonymizes PII</li>
    </ul>
  </li>
  <li><b>Output Generation:</b> User downloads the cleansed file or a text summary</li>
</ol>

<hr>

<h2>💡 Challenges & Solutions</h2>
<table>
  <thead>
    <tr>
      <th>Challenge</th>
      <th>Solution</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Handling multiple file formats</td><td>Unified parsing with dedicated extractors</td></tr>
    <tr><td>Accurate PII detection</td><td>Multi-layer NER via Presidio + regex</td></tr>
    <tr><td>Extracting text from complex images</td><td>Used Docling for OCR-based image parsing</td></tr>
    <tr><td>Balancing privacy & usability</td><td>Masking, hashing, and pseudonymization options</td></tr>
    <tr><td>Scalable deployment</td><td>Flask backend API with modular architecture</td></tr>
  </tbody>
</table>

<hr>

<h2>👨‍💻 Contributors</h2>
<ul>
  <li><b>Shivadev Manojkumar</b> </li>
  <li><b>Abhineeth Anoop</b> </li>
  <li><b>Kanhaiya Kumar</b> </li>
  <li><b>Maddipatla Chanukya Sai</b> </li>
  <li><b>Rishika Gondle</b> </li>
</ul>

<hr>
