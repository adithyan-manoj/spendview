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

// Data storage
let dataItems = [];

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    updateEnterButtonState();
    setupEventListeners();
});

function setupEventListeners() {
    // Camera button - opens file dialog
    cameraBtn.addEventListener("click", () => {
        billInput.click();
    });

    // File input change - when image is selected
    billInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            addDataItem("📷 Image selected: " + e.target.files[0].name, "camera");
            showNotification("Image selected");
        }
    });

    // Mic button
    micBtn.addEventListener("click", () => {
        addDataItem("🎤 Voice recording started", "voice");
        showNotification("Microphone clicked");
    });

    // Enter button - submits the form
    enterBtn.addEventListener("click", submitData);

    // Clear button
    clearBtn.addEventListener("click", clearAllData);

    // Input field events
    dataInput.addEventListener("input", updateEnterButtonState);
    dataInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") submitData();
    });
}

function submitData() {
    const question = dataInput.value.trim();
    
    if (!question) {
        addDataItem("❌ Please enter a question", "error");
        return;
    }

    // Add user's question to display
    addDataItem(question, "user");

    // Show loading state
    addDataItem("⏳ Processing your request...", "system");

    const formData = new FormData();
    if (billInput.files.length > 0) {
        formData.append("bill", billInput.files[0]);
    }
    formData.append("question", question);

    fetch("/que", {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading message
        dataItems = dataItems.filter(item => item.content !== "⏳ Processing your request...");
        
        if (data.success) {
            // Add Gemini response
            addDataItem(data.answer, "gemini");
        } else {
            addDataItem(`❌ Error: ${data.error}`, "error");
        }
    })
    .catch(error => {
        addDataItem(`❌ Network Error: ${error.message}`, "error");
    })
    .finally(() => {
        // Clear input fields
        dataInput.value = "";
        billInput.value = "";
        updateEnterButtonState();
    });
}

function addDataItem(content, type = "manual") {
    const timestamp = new Date().toLocaleTimeString();
    const item = {
        id: Date.now(),
        content: content,
        type: type,
        timestamp: timestamp,
    };

    dataItems.unshift(item);
    updateDisplay();
}

function updateDisplay() {
    if (dataItems.length === 0) {
        displayContent.innerHTML = `
            <div class="welcome-message">
                <i class="fas fa-database"></i>
                <p>Enter data below to see it displayed here</p>
            </div>
        `;
    } else {
        displayContent.innerHTML = dataItems.map(item => `
            <div class="data-item ${item.type}">
                <div class="timestamp">${item.timestamp}</div>
                <div class="content">${item.content}</div>
            </div>
        `).join("");
    }
}

function clearAllData() {
    if (dataItems.length > 0) {
        dataItems = [];
        updateDisplay();
        showNotification("All data cleared");
    }
}

function updateEnterButtonState() {
    if (dataInput.value.trim() !== "") {
        enterBtn.disabled = false;
        enterBtn.classList.add("has-content");
    } else {
        enterBtn.disabled = true;
        enterBtn.classList.remove("has-content");
    }
}

function showNotification(message) {
    // Simple notification implementation
    const notification = document.createElement("div");
    notification.className = "notification";
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add("fade-out");
        setTimeout(() => notification.remove(), 500);
    }, 2000);
}