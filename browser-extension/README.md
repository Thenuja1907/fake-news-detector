# Truth Guardian - Browser Extension

## Installation
1. Open Google Chrome.
2. Navigate to `chrome://extensions/`.
3. Enable **Developer mode** (top right toggle).
4. Click **Load unpacked**.
5. Select the `browser-extension` folder in this project (`.../fake-news-detector/browser-extension`).
6. The extension icon (Shield) should appear in your toolbar.

## Features
- **Real-time Page Analysis**: Click the extension icon to analyze the current page content.
- **Context Menu**: Highlight any text on any webpage, right-click, and select "🛡️ Analyze Verification for selection".
- **Premium UI**: Glassmorphism tooltips and sleek dark-mode popup.
- **Dashboard Integration**: Links directly to your User Dashboard.

## Troubleshooting
- If analysis fails, ensure the Backend Server is running (`python backend/run.py`).
- Check the extension console for errors (Right click icon -> Inspect Popup).
