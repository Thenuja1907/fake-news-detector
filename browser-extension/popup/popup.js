document.addEventListener('DOMContentLoaded', () => {
  const analyzeBtn = document.getElementById('analyze-btn');
  const statusEl = document.getElementById('status');
  const resultCard = document.getElementById('result');

  // UI Elements
  const classificationEl = document.getElementById('classification');
  const scoreEl = document.getElementById('score');
  const sentimentEl = document.getElementById('sentiment');
  const explanationEl = document.getElementById('explanation');

  analyzeBtn.addEventListener('click', async () => {
    // Reset UI
    resultCard.classList.add('hidden');
    statusEl.textContent = "Scanning page content...";
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";

    try {
      // 1. Get Active Tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab) {
        throw new Error("No active tab found");
      }

      // 2. Extract Text from Page
      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          // Simple text extraction heuristics
          return document.body.innerText.substring(0, 5000); // Limit to 5k chars for performance
        }
      });

      const pageText = results[0].result;
      const pageUrl = tab.url;

      statusEl.textContent = "Processing with AI...";

      // 3. Call Backend API
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: pageText,
          url: pageUrl
        })
      });

      if (!response.ok) {
        throw new Error("Server error");
      }

      const data = await response.json();

      // 4. Update UI
      displayResults(data);

    } catch (error) {
      statusEl.textContent = "Error: " + error.message;
      statusEl.style.color = "#ef4444";
      console.error(error);
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Current Page";
    }
  });

  function displayResults(data) {
    statusEl.textContent = "Analysis Complete";
    resultCard.classList.remove('hidden');

    // Classification Color
    classificationEl.textContent = data.classification;
    classificationEl.className = "value"; // reset
    if (data.classification === "Fake News") {
      classificationEl.classList.add("verdict-fake");
    } else if (data.classification === "Real News") {
      classificationEl.classList.add("verdict-real");
    }

    scoreEl.textContent = data.credibility_score + "%";
    sentimentEl.textContent = data.sentiment;
    explanationEl.textContent = data.explanation;
  }
});
