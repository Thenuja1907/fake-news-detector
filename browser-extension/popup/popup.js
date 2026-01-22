document.addEventListener("DOMContentLoaded", function () {
  const analyzeBtn = document.getElementById("analyze-btn");
  const statusText = document.getElementById("status");
  const resultDiv = document.getElementById("result");
  const classificationEl = document.getElementById("classification");
  const scoreEl = document.getElementById("score");
  const sentimentEl = document.getElementById("sentiment");
  const explanationEl = document.getElementById("explanation");

  // Check if we are on a valid page
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (tabs.length === 0) return;
    const activeTab = tabs[0];

    analyzeBtn.addEventListener("click", async () => {
      statusText.innerText = "Analyzing content...";
      resultDiv.classList.add("hidden");

      // 1. Scripting: Get text from the page
      try {
        const injectionResults = await chrome.scripting.executeScript({
          target: { tabId: activeTab.id },
          func: () => document.body.innerText // Simple full-text extraction
        });

        const pageContent = injectionResults[0].result;
        const pageUrl = activeTab.url;

        if (!pageContent || pageContent.length < 50) {
          statusText.innerText = "Not enough content to analyze.";
          return;
        }

        // 2. Call the Backend API
        const response = await fetch("http://localhost:5000/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content: pageContent, url: pageUrl })
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();

        // 3. Update UI
        statusText.innerText = "Analysis Complete";
        resultDiv.classList.remove("hidden");

        classificationEl.innerText = data.classification;
        scoreEl.innerText = data.credibility_score + "%";
        sentimentEl.innerText = data.sentiment;
        explanationEl.innerText = data.explanation;

        // Color coding
        if (data.classification === "Fake News") {
          classificationEl.style.color = "red";
        } else {
          classificationEl.style.color = "green";
        }

      } catch (err) {
        console.error(err);
        statusText.innerText = "Error: Could not connect to server.";
      }
    });
  });
});
