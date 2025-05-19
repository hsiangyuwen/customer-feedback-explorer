import React, { useState, FormEvent, ChangeEvent, useRef } from 'react';
import { UploadCloud, Loader2 } from 'lucide-react';

interface DataIngestionComponentProps {
  apiBaseUrl: string;
  onIngestionStatus: (type: 'success' | 'error' | 'info', message: string) => void;
}

const DataIngestionComponent: React.FC<DataIngestionComponentProps> = ({ apiBaseUrl, onIngestionStatus }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    } else {
      setSelectedFile(null);
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile) {
      onIngestionStatus('error', 'Please select a CSV file first.');
      return;
    }

    setIsUploading(true);
    onIngestionStatus('info', `Uploading ${selectedFile.name}...`);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${apiBaseUrl}/ingest/`, {
        method: 'POST',
        body: formData,
      });

      if (response.status === 202) { // Backend accepts the request
        const data = await response.json();
        onIngestionStatus('success', `File '${selectedFile.name}' upload accepted. Processing started: ${data.message || 'Check backend logs for progress.'}`);
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to parse error response from ingestion API.' }));
        onIngestionStatus('error', `Ingestion failed (Status ${response.status}): ${errorData.detail || response.statusText}`);
      }
    } catch (error) {
      console.error('Ingestion API error:', error);
      onIngestionStatus('error', `An error occurred during upload: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsUploading(false);
      setSelectedFile(null); // Clear selection after attempt
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // Reset file input visually
      }
    }
  };

  return (
    <div className="mb-8 p-6 bg-gray-800 rounded-xl shadow-2xl">
      <h2 className="text-2xl font-semibold mb-6 text-sky-400 flex items-center">
        <UploadCloud size={28} className="mr-3" />
        Ingest Feedback Data
      </h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="file-upload" className="block text-sm font-medium text-gray-300 mb-1">
            Upload CSV File
          </label>
          <div className="mt-1 flex items-center space-x-3">
            <input
              id="file-upload"
              ref={fileInputRef}
              name="file-upload"
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-sky-600 file:text-white hover:file:bg-sky-700 cursor-pointer"
            />
          </div>
           {selectedFile && <p className="text-xs text-gray-400 mt-2">Selected: {selectedFile.name}</p>}
        </div>
        <button
          type="submit"
          disabled={isUploading || !selectedFile}
          className="w-full flex items-center justify-center px-6 py-3 border border-transparent rounded-lg shadow-sm text-base font-medium text-white bg-sky-600 hover:bg-sky-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-sky-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors duration-150"
        >
          {isUploading ? (
            <>
              <Loader2 size={20} className="animate-spin mr-2" /> Processing...
            </>
          ) : (
            <>
              <UploadCloud size={20} className="mr-2" /> Process and Ingest
            </>
          )}
        </button>
      </form>
    </div>
  );
};

export default DataIngestionComponent;