const API_BASE = 'http://localhost:8000';

// Upload form handler
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const collectionInput = document.getElementById('collectionInput');
    const statusDiv = document.getElementById('uploadStatus');

    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const url = collectionInput.value
        ? `${API_BASE}/ingest?collection_name=${encodeURIComponent(collectionInput.value)}`
        : `${API_BASE}/ingest`;

    try {
        statusDiv.textContent = 'Uploading...';
        statusDiv.className = '';

        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const data = await response.json();
        statusDiv.textContent = `✓ Success! ${data.chunks_processed} chunks processed from "${data.pdf_name}"`;
        statusDiv.className = 'success';

        fileInput.value = '';
        loadStats();
    } catch (error) {
        statusDiv.textContent = `✗ Error: ${error.message}`;
        statusDiv.className = 'error';
    }
});

// Chat form handler
document.getElementById('chatForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const input = document.getElementById('messageInput');
    const chatBox = document.getElementById('chatBox');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to chat
    const userMsg = document.createElement('div');
    userMsg.className = 'message user';
    userMsg.textContent = message;
    chatBox.appendChild(userMsg);

    input.value = '';
    chatBox.scrollTop = chatBox.scrollHeight;

    // Add loading indicator
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'message loading';
    loadingMsg.textContent = 'Thinking...';
    chatBox.appendChild(loadingMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: message })
        });

        if (!response.ok) {
            throw new Error(`Request failed: ${response.statusText}`);
        }

        const data = await response.json();

        // Remove loading message
        chatBox.removeChild(loadingMsg);

        // Add assistant response
        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'message assistant';
        assistantMsg.innerHTML = marked.parse(data.answer);
        chatBox.appendChild(assistantMsg);
    } catch (error) {
        chatBox.removeChild(loadingMsg);
        const errorMsg = document.createElement('div');
        errorMsg.className = 'message assistant';
        errorMsg.textContent = `Error: ${error.message}`;
        chatBox.appendChild(errorMsg);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
});

// Load stats
async function loadStats() {
    const statsContainer = document.getElementById('statsContainer');

    try {
        const response = await fetch(`${API_BASE}/visualize/stats`);

        if (!response.ok) {
            throw new Error('Failed to load stats');
        }

        const data = await response.json();

        let html = `<div class="stat-item">
            <strong>Total Documents:</strong> ${data.total_documents}
        </div>`;

        if (Object.keys(data.pdfs).length > 0) {
            html += '<div class="stat-item"><strong>PDFs:</strong>';
            for (const [pdfName, stats] of Object.entries(data.pdfs)) {
                html += `<br/>• ${pdfName}: ${stats.chunks} chunks (${stats.total_chars} chars)`;
            }
            html += '</div>';
        }

        statsContainer.innerHTML = html || '<p>No documents yet</p>';
    } catch (error) {
        statsContainer.innerHTML = `<p style="color: red;">Error loading stats: ${error.message}</p>`;
    }
}

// Load stats on page load
window.addEventListener('load', loadStats);
