const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const statusCard = document.getElementById("status-card");
const statusText = document.getElementById("status-text");
const progressBar = document.getElementById("progress-bar");
const progressMeta = document.getElementById("progress-meta");
const submitButton = document.getElementById("submit-button");
const resultSection = document.getElementById("result");
const transcriptOutput = document.getElementById("transcript-output");
const downloadLink = document.getElementById("download-link");

function setStatus(message, tone = "muted") {
  statusText.textContent = message;
  statusCard.className = `status-card ${tone}`;
}

function setProgress(value, label = null) {
  const percent = Math.max(0, Math.min(100, Math.round(value * 100)));
  progressBar.style.width = `${percent}%`;
  progressMeta.textContent = label ? `${percent}% · ${label}` : `${percent}%`;
}

function renderSegments(segments) {
  return segments.map(([start, end, text, speaker]) => {
    const prefix = speaker ? `${speaker}: ` : "";
    return `[${start.toFixed(2)}s -> ${end.toFixed(2)}s] ${prefix}${text}`;
  }).join("\n");
}

function renderTranscript(payload) {
  if (payload.preview_text) {
    return payload.preview_text;
  }
  return renderSegments(payload.segments || []);
}

function attachFile(file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setStatus(`Ready to transcribe ${file.name}.`);
  setProgress(0);
}

async function pollJob(jobId) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Failed to load job status");
    }

    setStatus(payload.message || "Processing...");
    setProgress(payload.progress || 0, payload.status || "running");

    if (payload.status === "completed") {
      return payload.result;
    }

    if (payload.status === "failed") {
      throw new Error(payload.error || "Transcription failed");
    }

    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
}

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) {
    attachFile(file);
  }
});

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file) {
    setStatus(`Ready to transcribe ${file.name}.`);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const [file] = fileInput.files;
  if (!file) {
    setStatus("Choose a file first.", "error");
    return;
  }

  submitButton.disabled = true;
  setStatus(`Uploading ${file.name}...`);
  setProgress(0.02, "uploading");
  resultSection.classList.add("hidden");
  downloadLink.hidden = true;

  const data = new FormData(form);
  // An unchecked box is left out of FormData, which would let the server's
  // INCLUDE_DIARIZATION default apply; always send the user's choice.
  data.set("diarize", document.getElementById("diarize").checked ? "true" : "false");

  try {
    const payload = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/transcribe");
      xhr.responseType = "json";

      xhr.upload.addEventListener("progress", (progressEvent) => {
        if (!progressEvent.lengthComputable) {
          return;
        }
        const uploadProgress = (progressEvent.loaded / progressEvent.total) * 0.12;
        setStatus(`Uploading ${file.name}...`);
        setProgress(uploadProgress, "uploading");
      });

      xhr.addEventListener("load", async () => {
        const response = xhr.response || {};
        if (xhr.status < 200 || xhr.status >= 300) {
          reject(new Error(response.error || "Request failed"));
          return;
        }

        if (!response.job_id) {
          resolve(response);
          return;
        }

        try {
          setStatus(response.message || "Upload complete");
          setProgress(response.progress || 0.12, response.status || "queued");
          const jobResult = await pollJob(response.job_id);
          resolve(jobResult);
        } catch (error) {
          reject(error);
        }
      });

      xhr.addEventListener("error", () => {
        reject(new Error("Network error while uploading file"));
      });

      xhr.send(data);
    });

    transcriptOutput.textContent = renderTranscript(payload);
    resultSection.classList.remove("hidden");
    if (payload.download_url) {
      downloadLink.href = payload.download_url;
      downloadLink.hidden = false;
    }
    setProgress(1, "completed");
    setStatus(`Completed in ${payload.processing_time.toFixed(1)} seconds.`, "success");
  } catch (error) {
    setProgress(0, "error");
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});

fetch("/api/status")
  .then((response) => response.json())
  .then((payload) => {
    const info = payload.model || {};
    const model = info.model || info.model_size || "unknown";
    const engine = info.engine ? `${info.engine} ` : "";
    setStatus(`Server ready. Model: ${engine}${model}.`);
    setProgress(0);
  })
  .catch(() => {
    setStatus("Server status unavailable. Start the model server first.", "error");
    setProgress(0, "offline");
  });
