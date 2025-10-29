# GenAI-3-05

This repository contains the solution for the course **"Introduction to Project Activities"**.  
It belongs to the second block of assignments — **GenAI-3-05**.

---

## SEO Content System

A Python script that uses a **local LLM** (via [Ollama](https://ollama.com))  
to automatically generate SEO-optimized website content, including titles, meta descriptions, and short summaries.

---

## What it does

- Generates **SEO Title**, **Meta Description**, and **Summary**
- Automatically detects the language (Russian / English)
- Produces synthetic keywords if none are provided
- Validates text length and keyword inclusion
- Saves results as a JSON report
- Runs fully **offline** — no OpenAI or Hugging Face required

---

## Key Features

- **Local LLM Integration:** Works through Ollama — no VPN or API key needed  
- **Keyword Intelligence:** Automatically generates relevant SEO keywords  
- **Language Detection:** Supports both Russian and English  
- **Quality Control:** Checks content length and keyword coverage  
- **JSON Export:** Saves structured results for later analysis  
- **Offline Operation:** 100% local generation  

---

## Installation

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a compatible model
ollama pull qwen2.5:3b-instruct

# 3. Clone the repository and set up the environment
git clone https://github.
