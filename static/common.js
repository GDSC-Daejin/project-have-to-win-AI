// common.js

// Drag and drop functionality
var dropArea = document.getElementById('drop-area');

dropArea.addEventListener('dragenter', handleDragEnter, false);
dropArea.addEventListener('dragleave', handleDragLeave, false);
dropArea.addEventListener('dragover', handleDragOver, false);
dropArea.addEventListener('drop', handleFileSelect, false);

function handleDragEnter(e) {
    e.preventDefault();
    dropArea.classList.add('highlight');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropArea.classList.remove('highlight');
}

function handleDragOver(e) {
    e.preventDefault();
}

function handleFileSelect(e) {
    e.preventDefault();
    dropArea.classList.remove('highlight');
    var files = e.dataTransfer.files;
    document.getElementById('fileInput').files = files;
}

// EventSource and log handling
function setupEventSource() {
    const eventSource = new EventSource("/stream_logs");
    const logContainer = document.getElementById("log-container");

    eventSource.onmessage = function(event) {
        const logMessage = event.data;
        logContainer.innerHTML += logMessage;
        logContainer.scrollTop = logContainer.scrollHeight;
    };

    eventSource.onerror = function(event) {
        eventSource.close();
    };
}

// jQuery and AJAX functionality
function stopTrain(url) {
    $.ajax({
        type: 'POST',
        url: url,
        success: function(response) {
            alert("Training stopped successfully!");
        }
    });
}

// Custom option handling
function checkCustomOption() {
    var modelSelect = document.getElementById("model");
    var resumeInput = document.getElementById("resumeInput");
    var transferInput = document.getElementById("transferInput");
    var params = document.getElementById("params");

    if (modelSelect.value === "transfer_mode") {
        transferInput.style.display = "block";
        resumeInput.style.display = "none";
        params.style.display = "block";
    } else if (modelSelect.value === "resume_mode") {
        transferInput.style.display = "none";
        resumeInput.style.display = "block";
        params.style.display = "none";
    } else {
        transferInput.style.display = "none";
        resumeInput.style.display = "none";
        params.style.display = "block";
    }
}

// Set custom value for form submission
function setCustomValue() {
    var modelSelect = document.getElementById("model");

    if (modelSelect.value === "resume_mode") {
        var resumeInput = document.getElementById("resumePath");
        modelSelect.options[modelSelect.selectedIndex].value = resumeInput.value;
    } else if (modelSelect.value === "transfer_mode") {
        var transferInput = document.getElementById("transferPath");
        modelSelect.options[modelSelect.selectedIndex].value = transferInput.value;
    }
}
