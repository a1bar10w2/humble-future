# Humble Future

Free SAT and IELTS preparation: question bank, full practice tests, vocabulary, a daily challenge and an AI tutor.

Runs on Cloudflare Workers (free plan):

- `public/index.html` is the whole site.
- `src/worker.js` answers `/api/*` and runs the AI with Cloudflare Workers AI (no API key needed).
- Optional: add a secret `GEMINI_API_KEY` in the Worker settings to use Google Gemini first.

Student progress is saved in each student's own browser. No accounts, no personal data on the server.
