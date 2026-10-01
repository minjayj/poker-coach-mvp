# Poker Coach MVP

Build a local-first poker home-game assistant.

Important constraints:
- This is for a friendly home game where all players consent.
- Do not build cheating, stealth, casino, or covert-use features.
- MVP should be passive by default: camera + microphone detect events, user only corrects mistakes.
- Prefer simple working code over complex AI.
- Use algorithms for poker decisions; use LLM only for speech parsing and explanations.

Tech stack:
- Backend: Python FastAPI
- Realtime: WebSockets
- Storage: JSON first, SQLite later
- Speech: Whisper or pluggable speech-to-text interface
- Vision: OpenCV placeholder first, real card detection later
- Frontend: simple browser UI, mobile-friendly

MVP features:
1. Player/seat configuration
2. Live hand state
3. Speech action parser
4. Camera frame ingestion placeholder
5. Player profile JSON files
6. Manual correction buttons
7. Basic recommendation engine
8. Session/action logging

Do not overbuild:
- No facial recognition in MVP
- No chip-counting in MVP
- No native mobile app in MVP
- No solver integration in MVP

AGENTS.md is commonly used to give coding agents repo-specific guidance.
