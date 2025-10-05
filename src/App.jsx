import React, { useState, useRef } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [uploadId, setUploadId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [preserveType, setPreserveType] = useState(false);
  const progressInterval = useRef(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setProgress(0);
    setDownloadUrl(null);
    setUploadId(null);
    const form = new FormData();
    form.append("file", file);
    // Add preserveType flag if relevant
    if (preserveType && (file.name.endsWith('.csv') || file.name.endsWith('.xlsx') || file.name.endsWith('.pdf') || 
        file.name.endsWith('.jpg') || file.name.endsWith('.jpeg') || file.name.endsWith('.png'))) {
      form.append("preserve_type", "true");
    }
    try {
      const res = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      if (data.upload_id && data.download_url) {
        setUploadId(data.upload_id);
        setDownloadUrl(data.download_url.replace("/download/", ""));
        // Start polling progress
        progressInterval.current = setInterval(async () => {
          try {
            const resp = await fetch(`http://localhost:5000/progress/${data.upload_id}`);
            const prog = await resp.json();
            setProgress(prog.progress);
            if (prog.progress >= 100) {
              clearInterval(progressInterval.current);
              setLoading(false);
            } else if (prog.progress < 0) {
              clearInterval(progressInterval.current);
              setLoading(false);
              alert("Processing failed.");
            }
          } catch {
            clearInterval(progressInterval.current);
            setLoading(false);
            alert("Error checking progress.");
          }
        }, 1000);
      } else {
        setLoading(false);
        alert("Upload failed.");
      }
    } catch (err) {
      setLoading(false);
      alert("Upload failed");
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4 text-slate-800 text-center uppercase">
        FILE CLEANSING & ANALYSIS
      </h1>
      <p className="mb-4 text-slate-600 text-center">
        Upload a file (PDF, Word, Excel, Image, PPT, or Text). The system will
        cleanse PII and analyze for security-related insights.
      </p>

      {/* Drag & Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer ${
          dragOver ? "border-blue-500 bg-blue-50" : "border-gray-300"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById("fileInput").click()}
      >
        <input
          id="fileInput"
          type="file"
          accept=".pdf,.docx,.xlsx,.xlsm,.pptx,.png,.jpg,.jpeg,.bmp,.tiff,.txt,.csv"
          className="hidden"
          onChange={(e) => setFile(e.target.files[0])}
        />
        {file ? (
          <p className="text-slate-700">Selected: {file.name}</p>
        ) : (
          <p className="text-slate-500">
            Drag & Drop your file here, or{" "}
            <span className="text-blue-600">browse</span>
          </p>
        )}
      </div>

      {/* Upload Button */}
      {/* Preserve file type checkbox for CSV/XLSX/PDF/Images */}
      {file && (file.name.endsWith('.csv') || file.name.endsWith('.xlsx') || file.name.endsWith('.pdf') || 
                file.name.endsWith('.jpg') || file.name.endsWith('.jpeg') || file.name.endsWith('.png')) && (
        <div className="mt-4 flex items-center justify-center">
          <input
            type="checkbox"
            id="preserveType"
            checked={preserveType}
            onChange={e => setPreserveType(e.target.checked)}
            className="mr-2"
          />
          <label htmlFor="preserveType" className="text-slate-700 font-medium">
            Preserve file type
          </label>
        </div>
      )}

      <div className="mt-4 flex justify-center">
        <button
          className="px-6 py-3 rounded-lg bg-red-600 hover:bg-red-700 text-white font-semibold text-lg disabled:opacity-50 whitespace-nowrap"
          type="submit"
          onClick={handleSubmit}
          disabled={loading || !file}
        >
          {loading ? "Processing..." : "Upload & Run"}
        </button>
      </div>

      {/* Progress Bar */}
      {loading && (
        <div className="mt-6">
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div
              className="bg-blue-600 h-4 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="text-center mt-2 text-sm text-slate-600">{progress}%</div>
        </div>
      )}

      {/* Download Button */}
      {progress === 100 && downloadUrl && (
        <div className="mt-8 flex flex-col items-center">
          <a
            href={`http://localhost:5000/download/${downloadUrl}`}
            className="px-6 py-3 rounded-lg bg-green-600 hover:bg-green-700 text-white font-semibold text-lg"
            download
          >
            Download Cleansed File
          </a>
        </div>
      )}
    </div>
  );
}
