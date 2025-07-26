// Get DOM elements
const cameraBtn = document.getElementById("cameraBtn");
const enterBtn = document.getElementById("enterBtn");
const micBtn = document.getElementById("micBtn");
const dataInput = document.getElementById("dataInput");
const displayContent = document.getElementById("displayContent");
const clearBtn = document.getElementById("clearBtn");
const welcomeMessage = document.getElementById("welcomeMessage");
const billInput = document.getElementById("billInput");
const dataForm = document.getElementById("dataForm");
const micIcon = document.getElementById("micIcon");
const stopBtn = document.getElementById("stopBtn");

// Data storage
let dataItems = [];

// Audio recording variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordingInterval;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    updateEnterButtonState();
    setupEventListeners();
});

function setupEventListeners() {
    dataForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitData();
    });

    cameraBtn.addEventListener("click", () => {
        billInput.click();
    });

    billInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            addDataItem("📷 Image selected: " + e.target.files[0].name, "camera");
            showNotification("Image selected");
        }
    });

    micBtn.addEventListener("click", startRecording);
    stopBtn.addEventListener("click", stopRecording);
    enterBtn.addEventListener("click", submitData);
    clearBtn.addEventListener("click", clearAllData);
    dataInput.addEventListener("input", updateEnterButtonState);

    // --- Multiline input logic for textarea ---
    dataInput.addEventListener("keydown", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitData();
        }
        // Shift+Enter will insert a newline by default
    });

    dataInput.addEventListener("input", function () {
        // Auto-resize textarea up to max-height (120px)
        this.style.height = "40px";
        this.style.height = Math.min(this.scrollHeight, 120) + "px";
    });
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioToServer(audioBlob);
            audioChunks = [];
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        micBtn.style.display = 'none';
        stopBtn.style.display = 'flex';
        micIcon.className = 'fas fa-microphone-slash';
        showNotification("Recording started...");
        
        // Start collecting data periodically
        recordingInterval = setInterval(() => {
            if (isRecording) {
                mediaRecorder.requestData();
            }
        }, 1000);
        
    } catch (error) {
        console.error("Error accessing microphone:", error);
        addDataItem("❌ Microphone access denied: " + error.message, "error");
        showNotification("Could not access microphone");
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        clearInterval(recordingInterval);
        mediaRecorder.stop();
        isRecording = false;
        
        // Stop all tracks in the stream
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        // Update UI
        micBtn.style.display = 'flex';
        stopBtn.style.display = 'none';
        micIcon.className = 'fas fa-microphone';
    }
}

async function sendAudioToServer(audioBlob) {
    addDataItem("🎤 Processing recorded audio...", "system");
    
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.wav");
    
    // Include text input if available
    const question = dataInput.value.trim();
    if (question) {
        formData.append("question", question);
    }
    if (billInput.files.length > 0) {
        formData.append("bill", billInput.files[0]);
    }
    try {
        const response = await fetch("/que", {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Network response was not ok');
        
        const data = await response.json();
        dataItems = dataItems.filter(item => item.content !== "🎤 Processing recorded audio...");
        
        if (data.success) {
            addDataItem(data.answer, "gemini");
        } else {
            addDataItem(`❌ Error: ${data.error}`, "error");
        }
    } catch (error) {
        dataItems = dataItems.filter(item => item.content !== "🎤 Processing recorded audio...");
        addDataItem(`❌ Network Error: ${error.message}`, "error");
    } finally {
        dataInput.value = "";
        billInput.value = "";
        updateEnterButtonState();
    }
}

function submitData() {
    const question = dataInput.value.trim();

    if (!question && billInput.files.length === 0) {
        if (!dataItems.some(item => item.content === "❌ Please enter a question or upload a file")) {
            addDataItem("❌ Please enter a question or upload a file", "error");
        }
        return;
    }

    if (question) {
        addDataItem(question, "user");
    }
    if (billInput.files.length > 0) {
        addDataItem("📷 Image uploaded for analysis", "system");
    }
    addDataItem("⏳ Processing your request...", "system");

    const formData = new FormData();
    if (question) {
        formData.append("question", question);
    }

    // Only append the image if one was selected
    if (billInput.files.length > 0) {
        formData.append("bill", billInput.files[0]);
    }

    fetch("/que", {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    })
    .then(data => {
        dataItems = dataItems.filter(item => item.content !== "⏳ Processing your request..."&&
            item.content !== "📷 Image uploaded for analysis");
        if (data.success) {
            addDataItem(data.answer, "gemini");
        } else {
            addDataItem(`❌ Error: ${data.error}`, "error");
        }
    })
    .catch(error => {
        dataItems = dataItems.filter(item => item.content !== "⏳ Processing your request..."&&
            item.content !== "📷 Image uploaded for analysis");
        addDataItem(`❌ Network Error: ${error.message}`, "error");
    })
    .finally(() => {
        dataInput.value = "";
        dataInput.style.height = "40px"; // Reset textarea height after submit
        billInput.value = "";
        updateEnterButtonState();
    });
}

function addDataItem(content, type = "manual") {
    const timestamp = new Date().toLocaleTimeString();
    let formattedContent = content;
    
    if (type === 'gemini') {
        formattedContent = formatGeminiResponse(content);
    }

    const item = {
        id: Date.now(),
        content: formattedContent,
        type: type,
        timestamp: timestamp,
    };

    dataItems.unshift(item);
    updateDisplay();
}

function formatGeminiResponse(text) {
    // Check if this is a detailed bill
    if (isDetailedBill(text)) {
        return formatDetailedBill(text);
    }
    
    // Check if this contains a markdown table
    if (containsMarkdownTable(text)) {
        return formatMarkdownTable(text);
    }
    
    // Default formatting
    return formatAsText(text);
}

function isDetailedBill(text) {
    return text.includes("Restaurant Information:") && 
           text.includes("Bill Details:") && 
           text.includes("Itemized List:");
}

function formatDetailedBill(text) {
    const sections = {
        restaurant: extractSection(text, "Restaurant Information:", "Bill Details:"),
        bill: extractSection(text, "Bill Details:", "Itemized List:"),
        items: extractSection(text, "Itemized List:", "Summary:"),
        summary: extractSection(text, "Summary:", "Delivery Contact:") || 
                 extractSection(text, "Summary:", "")
    };

    let html = '<div class="bill-container">';
    
    // Restaurant Info
    html += `<div class="bill-section">
                <h3>Restaurant Information</h3>
                ${formatInfoLines(sections.restaurant)}
            </div>`;
    
    // Bill Details
    html += `<div class="bill-section">
                <h3>Bill Details</h3>
                ${formatInfoLines(sections.bill)}
            </div>`;
    
    // Items Table
    html += `<div class="bill-section">
                <h3>Items</h3>
                ${formatItemsTable(sections.items)}
            </div>`;
    
    // Summary
    html += `<div class="bill-section">
                <h3>Summary</h3>
                ${formatInfoLines(sections.summary)}
            </div>`;
    
    html += '</div>';
    return html;
}

function containsMarkdownTable(text) {
    return text.includes('|') && text.includes('-|-');
}

function formatMarkdownTable(text) {
    const lines = text.split('\n').filter(line => line.trim() && line.includes('|'));
    if (lines.length < 2) return formatAsText(text);
    
    const headers = lines[0].split('|').slice(1, -1).map(h => h.trim());
    const separatorIndex = lines.findIndex(line => line.startsWith('| :-'));
    const contentLines = separatorIndex >= 0 ? lines.slice(separatorIndex + 1) : lines.slice(1);
    
    let html = '<div class="table-container"><table class="data-table">';
    html += '<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead>';
    html += '<tbody>';
    
    contentLines.forEach(line => {
        const cells = line.split('|').slice(1, -1).map(c => c.trim());
        const isTotalRow = cells.some(cell => /total/i.test(cell));
        html += `<tr${isTotalRow ? ' class="total-row"' : ''}>`;
        cells.forEach((cell, i) => {
            const isNumber = i > 0 && /^[\d,.]+$/.test(cell);
            html += `<td${isNumber ? ' class="number-cell"' : ''}>${cell}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    return html;
}

function extractSection(text, startMarker, endMarker) {
    const start = text.indexOf(startMarker) + startMarker.length;
    const end = endMarker ? text.indexOf(endMarker) : text.length;
    return text.substring(start, end).trim();
}

function formatInfoLines(text) {
    return text.split('\n')
        .filter(line => line.trim())
        .map(line => {
            if (line.includes(':')) {
                const [label, value] = line.split(':').map(part => part.trim());
                return `<div class="info-line"><span class="label">${label}:</span> <span class="value">${value}</span></div>`;
            }
            return `<div class="info-line">${line}</div>`;
        })
        .join('');
}

function formatItemsTable(text) {
    const lines = text.split('\n').filter(line => line.trim() && line.includes('|'));
    if (lines.length < 2) return text;
    
    // Remove markdown table formatting line
    const dataLines = lines.filter(line => !line.startsWith('| :'));
    
    let html = '<div class="table-container"><table class="data-table">';
    
    // Headers
    const headers = dataLines[0].split('|').slice(1, -1).map(h => h.trim());
    html += '<thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>';
    
    // Rows
    for (let i = 1; i < dataLines.length; i++) {
        const cells = dataLines[i].split('|').slice(1, -1).map(c => c.trim());
        const isTotalRow = cells.some(cell => /total/i.test(cell));
        html += `<tr${isTotalRow ? ' class="total-row"' : ''}>`;
        cells.forEach((cell, j) => {
            const isNumber = j > 0 && /^[\d,.]+$/.test(cell);
            html += `<td${isNumber ? ' class="number-cell"' : ''}>${cell}</td>`;
        });
        html += '</tr>';
    }
    
    html += '</tbody></table></div>';
    return html;
}

function formatAsText(text) {
    let formattedText = text
        .replace(/^##\s+(.+)$/gm, '<h3>$1</h3>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^-\s+(.+)$/gm, '<li>$1</li>')
        .replace(/^\*\s+(.+)$/gm, '<li>$1</li>')
        .replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    if (!formattedText.includes('<h3>') && 
        !formattedText.includes('<li>') && 
        !formattedText.includes('<pre>') &&
        !formattedText.startsWith('<p>')) {
        formattedText = '<p>' + formattedText + '</p>';
    }

    formattedText = formattedText.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    return formattedText;
}

function updateDisplay() {
    displayContent.innerHTML = dataItems.length === 0
        ? `<div class="welcome-message">
              <i class="fas fa-database"></i>
              <p>Enter data below to see it displayed here</p>
           </div>`
        : dataItems.map(item => `
            <div class="data-item ${item.type}">
                <div class="timestamp">${item.timestamp}</div>
                <div class="content">${item.content}</div>
            </div>`
          ).join("");
}

function clearAllData() {
    if (isRecording) {
        stopRecording();
    }
    if (dataItems.length > 0) {
        dataItems = [];
        dataInput.value = "";
        billInput.value = "";
        updateEnterButtonState();
        updateDisplay();
        showNotification("All data cleared");
    }
}

function updateEnterButtonState() {
    const hasContent = dataInput.value.trim() !== "" || billInput.files.length > 0;
    enterBtn.disabled = !hasContent;
    enterBtn.classList.toggle("has-content", hasContent);
}

function showNotification(message) {
    const notification = document.createElement("div");
    notification.className = "notification";
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add("fade-out");
        setTimeout(() => notification.remove(), 500);
    }, 2000);
}